use std::sync::{Arc, Mutex};

use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use log::warn;

use crate::{
    buffer::Buffer,
    command::{CmdBufferGen, CommandRecorder},
    graphics_pipeline::{GraphicsAttach, GraphicsPipeline},
    image::{Image, ImageView},
    instance::{InstanceDropper, VkGpuInfo},
    shader::ShaderSet,
    swapchain::Swapchain,
    sync::{BinSem, CpuFuture, GpuFuture, SyncPool, TlSem, rhi2_pipe_stage_to_vk},
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
    pub allocator: Arc<Mutex<Allocator>>,
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
        let allocator_create_info = AllocatorCreateDesc {
            instance: instance_dropper.instance.clone(),
            device: device.clone(),
            physical_device: gpu_info.handle,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };
        let allocator = Allocator::new(&allocator_create_info)
            .map_err(|e| format!("creating gpu mem allocator failed: {e}"))?;
        Ok(Self {
            allocator: Arc::new(Mutex::new(allocator)),
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
    dropper: Arc<DeviceDropper>,
    swapchain: Swapchain,
    cmd_pool: CmdBufferGen,
    sync_pool: Arc<SyncPool>,
}

impl Device {
    pub fn new(instance_dropper: &Arc<InstanceDropper>, gpu_id: usize) -> Result<Self, String> {
        let dropper = DeviceDropper::new(instance_dropper, gpu_id)
            .map(Arc::new)
            .map_err(|e| format!("device dropper creation failed: {e}"))?;
        let sync_pool = Arc::new(SyncPool::new(&dropper));
        let swapchain = Swapchain::new(&dropper, &sync_pool, None)
            .map_err(|e| format!("swapchain create failed: {e}"))?;
        let cmd_pool = CmdBufferGen::new(&dropper);

        Ok(Self {
            dropper: dropper.clone(),
            swapchain,
            cmd_pool,
            sync_pool,
        })
    }
}

impl rhi2::device::Device for Device {
    type SC = Swapchain;

    type BType = Buffer;

    type IType = Image;

    type IVType = ImageView;

    type SSType = ShaderSet;

    type GAType = GraphicsAttach;

    type GPType = GraphicsPipeline;

    type CRType = CommandRecorder;

    type CFType = CpuFuture;

    type GFType = GpuFuture;

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
    ) -> Result<Self::BType, rhi2::device::DeviceErr> {
        Buffer::new(&self.dropper, size, flags, host_access)
            .map_err(rhi2::device::DeviceErr::BufferCreateFailed)
    }

    fn new_image(
        &self,
        format: rhi2::image::Format,
        res: (u32, u32, u32),
        layers: u32,
        flags: rhi2::enumflags2::BitFlags<rhi2::image::ImageFlags>,
        host_access: rhi2::HostAccess,
    ) -> Result<Self::IType, rhi2::device::DeviceErr> {
        Image::new(&self.dropper, format, res, layers, flags, host_access)
            .map_err(rhi2::device::DeviceErr::ImageCreateFailed)
    }

    fn new_graphics_pipeline(
        &self,
        shader: &str,
        sets: Vec<Vec<rhi2::shader::ShaderSetInfo>>,
        pc_size: usize,
        vert_stage_info: rhi2::graphics_pipeline::VertexStageInfo,
        frag_stage_info: rhi2::graphics_pipeline::FragmentStageInfo,
    ) -> Result<Self::GPType, rhi2::device::DeviceErr> {
        GraphicsPipeline::new(
            &self.dropper,
            shader,
            sets,
            pc_size,
            vert_stage_info,
            frag_stage_info,
        )
        .map_err(rhi2::device::DeviceErr::GraphicsPipelineCreateFailed)
    }

    fn new_cmd_recorders(
        &self,
        count: usize,
    ) -> Result<Vec<Self::CRType>, rhi2::device::DeviceErr> {
        CommandRecorder::new(&self.cmd_pool, &self.sync_pool, count)
            .map_err(|e| format!("cmd recorders creation failed: {e}"))
            .map_err(rhi2::device::DeviceErr::CmdRecorderCreateFailed)
    }
}

pub fn run_cmds_gpu_sync_internal(
    sync_pool: &Arc<SyncPool>,
    cbs: Vec<CommandRecorder>,
    wait_for: Vec<rhi2::command::CmdWaitInfo<GpuFuture>>,
    force_bin_sem: bool,
) -> Result<GpuFuture, String> {
    let mut gfut = if force_bin_sem {
        let bin_sem = BinSem::get(sync_pool, 1)
            .map_err(|e| format!("failed getting bin sem: {e}"))?
            .remove(0);
        GpuFuture::from_bin(bin_sem, vec![])
    } else {
        let tl_sem = TlSem::get(sync_pool, 1)
            .map_err(|e| format!("failed getting tl sem: {e}"))?
            .remove(0);
        GpuFuture::from_tl(tl_sem, vec![])
    };
    gfut.increase_count(1);

    let mut wait_sems = vec![];
    let mut wait_nums = vec![];
    let mut on_stages = vec![];
    let mut by_stages = vec![];
    for wait_info in wait_for {
        let Some((s, c)) = wait_info.gfut.get_wait_info() else {
            continue;
        };
        wait_sems.push(s);
        wait_nums.push(c);
        on_stages.push(rhi2_pipe_stage_to_vk(&wait_info.on));
        by_stages.push(rhi2_pipe_stage_to_vk(&wait_info.by));
        gfut.preserve_buffers
            .extend(wait_info.gfut.preserve_buffers);
    }

    let Some((sig_sem, sig_val)) = gfut.get_wait_info() else {
        unreachable!()
    };

    let mut tl_submit_info = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_nums)
        .signal_semaphore_values(core::slice::from_ref(&sig_val));
    let cb_vks: Vec<_> = cbs.iter().map(|c| c.inner.handle).collect();
    unsafe {
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cb_vks)
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&on_stages)
            .signal_semaphores(core::slice::from_ref(&sig_sem))
            .push_next(&mut tl_submit_info);
        sync_pool
            .device_dropper
            .device
            .queue_submit(
                sync_pool.device_dropper.gfx_queue,
                &[submit_info],
                vk::Fence::null(),
            )
            .map_err(|e| format!("queue submission error: {e}"))?;
    }
    Ok(gfut)
}
