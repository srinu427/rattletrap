use std::sync::Arc;

use ash::{ext, khr, vk};
use log::warn;
use rhi2::device::DeviceErr;

use crate::{
    buffer::Buffer,
    command::{CmdBufferGen, CommandRecorder},
    graphics_pipeline::GraphicsPipeline,
    image::{Image, ImageView},
    instance::{InstanceDropper, VkGpuInfo},
    memory::MemAlloc,
    shader::ShaderSet,
    swapchain::Swapchain,
    sync::{TaskFuture, TlSemPool},
};

fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        ext::descriptor_indexing::NAME.as_ptr(),
        // khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

pub struct DeviceDropper {
    pub swapchain_device: khr::swapchain::Device,
    pub gfx_queue: vk::Queue,
    pub device: ash::Device,
    pub gpu_info: VkGpuInfo,
    pub instance_dropper: Arc<InstanceDropper>,
}

impl DeviceDropper {
    pub fn new(instance_dropper: &Arc<InstanceDropper>, gpu_id: usize) -> Result<Self, String> {
        let gpu_info = instance_dropper.gpus[gpu_id].clone();

        let queue_priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gpu_info.gfx_qf as _)
            .queue_priorities(&queue_priorities)];
        let device_extensions = get_device_extensions();
        let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true);
        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features)
            .push_next(&mut device_12_features);
        let device = unsafe {
            instance_dropper
                .instance
                .create_device(gpu_info.handle, &device_create_info, None)
                .map_err(|e| format!("create vk device failed: {e}"))?
        };
        let gfx_queue = unsafe { device.get_device_queue(gpu_info.gfx_qf as _, 0) };
        let swapchain_device = khr::swapchain::Device::new(&instance_dropper.instance, &device);
        Ok(Self {
            swapchain_device,
            gfx_queue,
            device,
            gpu_info,
            instance_dropper: instance_dropper.clone(),
        })
    }
}

impl Drop for DeviceDropper {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.device.device_wait_idle() {
                warn!("error waiting for device to get idle before destroying: {e}")
            };
            self.device.destroy_device(None);
        }
    }
}

pub struct Device {
    pub dropper: Arc<DeviceDropper>,
    pub allocator: Arc<MemAlloc>,
    swapchain: Swapchain,
    cmd_pool: Arc<CmdBufferGen>,
    sync_pool: Arc<TlSemPool>,
}

impl Device {
    pub fn new(instance_dropper: &Arc<InstanceDropper>, gpu_id: usize) -> Result<Self, String> {
        let dropper = DeviceDropper::new(instance_dropper, gpu_id)
            .map(Arc::new)
            .map_err(|e| format!("device dropper creation failed: {e}"))?;
        let sync_pool = Arc::new(TlSemPool::new(&dropper));
        let cmd_pool = CmdBufferGen::new(&dropper)
            .map(Arc::new)
            .map_err(|e| format!("command pool creation failed: {e}"))?;
        let swapchain = Swapchain::new(&dropper, &cmd_pool, None)
            .map_err(|e| format!("swapchain create failed: {e}"))?;
        let allocator = MemAlloc::new(&dropper)
            .map(Arc::new)
            .map_err(|e| format!("mem allocator creation failed: {e}"))?;
        Ok(Self {
            dropper: dropper.clone(),
            allocator,
            swapchain,
            cmd_pool,
            sync_pool,
        })
    }
}

impl rhi2::device::Device for Device {
    type SC = Swapchain;

    type B = Buffer;

    type I = Image;

    type IV = ImageView;

    type SS = ShaderSet;

    type GP = GraphicsPipeline;

    type CR = CommandRecorder;

    type TF = TaskFuture;

    fn gpu_info(&self) -> rhi2::device::GpuInfo {
        let vk_gpu_info = &self.dropper.gpu_info;
        rhi2::device::GpuInfo {
            id: vk_gpu_info.id,
            name: vk_gpu_info.name.clone(),
            dvram: vk_gpu_info.dvram,
            is_dedicated: vk_gpu_info.is_dedicated,
        }
    }

    fn swapchain(&self) -> &Self::SC {
        &self.swapchain
    }

    fn swapchain_mut(&mut self) -> &mut Self::SC {
        &mut self.swapchain
    }

    fn new_buffer(
        &self,
        size: usize,
        flags: rhi2::enumflags2::BitFlags<rhi2::buffer::BufferFlags>,
        host_access: rhi2::HostAccess,
    ) -> Result<Self::B, DeviceErr> {
        Buffer::new(&self.allocator, size, flags, host_access)
            .map_err(DeviceErr::BufferCreateFailed)
    }

    fn new_image(
        &self,
        format: rhi2::image::Format,
        res: (u32, u32, u32),
        layers: u32,
        flags: rhi2::enumflags2::BitFlags<rhi2::image::ImageFlags>,
        host_access: rhi2::HostAccess,
    ) -> Result<Self::I, DeviceErr> {
        Image::new(&self.allocator, format, res, layers, flags, host_access)
            .map_err(DeviceErr::ImageCreateFailed)
    }

    fn new_graphics_pipeline(
        &self,
        shader: &str,
        sets: Vec<Vec<rhi2::shader::ShaderSetInfo>>,
        pc_size: usize,
        vert_stage_info: rhi2::graphics_pipeline::VertexStageInfo,
        frag_stage_info: rhi2::graphics_pipeline::FragmentStageInfo,
    ) -> Result<Self::GP, DeviceErr> {
        GraphicsPipeline::new(
            &self.dropper,
            shader,
            sets,
            pc_size,
            vert_stage_info,
            frag_stage_info,
        )
        .map_err(DeviceErr::GraphicsPipelineCreateFailed)
    }

    fn new_cmd_recorder(&self) -> Result<Self::CR, DeviceErr> {
        CommandRecorder::new(&self.cmd_pool, &self.sync_pool, 1)
            .map(|mut v| v.remove(0))
            .map_err(|e| format!("cmd recorder creation failed: {e}"))
            .map_err(DeviceErr::CmdRecorderCreateFailed)
    }

    fn run_work_graph(&self) -> Result<Self::TF, DeviceErr> {
        todo!()
    }

    fn wait_idle(&self) {
        unsafe {
            self.dropper
                .device
                .device_wait_idle()
                .inspect_err(|e| warn!("{e}"))
                .ok();
        }
    }
}
