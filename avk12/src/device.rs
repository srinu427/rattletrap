use std::sync::{Arc, Mutex};

use anyhow::Context;
use ash::{ext, khr, vk};
use getset::{Getters, MutGetters};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::{
    canvas::Canvas,
    pipeline::{GraphicsPipeline, GraphicsPipelineCreateInfo},
    resource::{BufferCreateInfo, BufferRef, ImageCreateInfo, ImageRef, Sampler},
    sync::SemPool,
    task::{CmdPool, Task},
};

#[derive(Debug, Clone, Getters)]
pub struct GpuInfo {
    #[getset(get = "pub")]
    id: usize,
    #[getset(get = "pub")]
    name: String,
    #[getset(get = "pub")]
    dvram: u64,
    #[getset(get = "pub")]
    is_dedicated: bool,
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) gfx_qf: usize,
}

fn get_instance_layers() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        c"VK_LAYER_KHRONOS_validation".as_ptr(),
    ]
}

fn get_instance_extensions() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        ext::debug_utils::NAME.as_ptr(),
        khr::get_physical_device_properties2::NAME.as_ptr(),
        khr::surface::NAME.as_ptr(),
        #[cfg(target_os = "windows")]
        khr::win32_surface::NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        khr::xlib_surface::NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        khr::wayland_surface::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_enumeration::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        ext::metal_surface::NAME.as_ptr(),
        #[cfg(target_os = "android")]
        khr::android_surface::NAME.as_ptr(),
    ]
}

pub fn create_instance(entry: &ash::Entry) -> anyhow::Result<ash::Instance> {
    let layers = get_instance_layers();
    let extensions = get_instance_extensions();
    let app_info = vk::ApplicationInfo::default()
        .api_version(vk::API_VERSION_1_2)
        .application_name(c"rattleapp")
        .application_version(0)
        .engine_name(c"rattletrap")
        .engine_version(0);
    #[cfg(target_os = "macos")]
    let create_info = vk::InstanceCreateInfo::default()
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    #[cfg(not(target_os = "macos"))]
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .context("instance creation failed")?
    };
    Ok(instance)
}

pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &Window,
) -> anyhow::Result<vk::SurfaceKHR> {
    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            window
                .display_handle()
                .context("get display handle failed")?
                .as_raw(),
            window
                .window_handle()
                .context("get window handle failed")?
                .as_raw(),
            None,
        )
        .context("surface creation failed")?
    };
    Ok(surface)
}

pub(crate) struct InstanceDropper {
    pub gpus: Vec<GpuInfo>,
    pub surface: vk::SurfaceKHR,
    pub surface_instance: khr::surface::Instance,
    pub instance: ash::Instance,
    pub window: Arc<Window>,
    _entry: ash::Entry,
}

impl InstanceDropper {
    fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load().context("vulkan load failed")? };
        let instance = create_instance(&entry).context("create instance failed")?;
        let surface =
            create_surface(&entry, &instance, &window).context("create surface failed")?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let gpus = unsafe {
            instance
                .enumerate_physical_devices()
                .context("getting gpu list failed")?
        };
        // Select GPUs supported by surface
        let mut gpu_dets = vec![];
        for gpu in gpus.into_iter() {
            let gpu_props = unsafe { instance.get_physical_device_properties(gpu) };
            let gpu_mem_props = unsafe { instance.get_physical_device_memory_properties(gpu) };
            let qf_props = unsafe { instance.get_physical_device_queue_family_properties(gpu) };
            let is_dedicated = gpu_props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            let gpu_name = gpu_props
                .device_name_as_c_str()
                .map(|cs| cs.to_string_lossy().to_string())
                .unwrap_or(format!("Unknown Device #{}", gpu_dets.len()));
            let dvram: u64 = gpu_mem_props
                .memory_heaps
                .iter()
                .filter(|mh| mh.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|mh| mh.size)
                .sum();
            if let Some((gfx_qf_id, _)) = qf_props
                .into_iter()
                .enumerate()
                .filter(|(i, _)| unsafe {
                    surface_instance
                        .get_physical_device_surface_support(gpu, *i as _, surface)
                        .unwrap_or(false)
                })
                .filter(|(_, qfp)| qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .max_by_key(|(_, qfp)| qfp.queue_count)
            {
                gpu_dets.push(GpuInfo {
                    id: gpu_dets.len(),
                    name: gpu_name,
                    dvram,
                    is_dedicated,
                    handle: gpu,
                    gfx_qf: gfx_qf_id,
                });
            }
        }
        Ok(Self {
            gpus: gpu_dets,
            surface,
            surface_instance,
            instance,
            window: window.clone(),
            _entry: entry,
        })
    }
}

pub struct Instance {
    pub(crate) dropper: Arc<InstanceDropper>,
}

impl Instance {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let dropper = InstanceDropper::new(window)
            .map(Arc::new)
            .context("instance dropper failed")?;
        Ok(Self { dropper })
    }

    pub fn get_gpus(&self) -> &[GpuInfo] {
        &self.dropper.gpus
    }

    pub fn init_device(self, gpu_id: usize) -> anyhow::Result<Device> {
        Device::new(&self.dropper, gpu_id).context("device creation failed")
    }
}

fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        ext::descriptor_indexing::NAME.as_ptr(),
        // khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

pub(crate) struct DeviceDropper {
    pub swapchain_device: khr::swapchain::Device,
    pub gfx_queue: vk::Queue,
    pub device: ash::Device,
    pub gpu_info: GpuInfo,
    pub instance_dropper: Arc<InstanceDropper>,
}

impl DeviceDropper {
    fn new(instance_dropper: &Arc<InstanceDropper>, gpu_id: usize) -> anyhow::Result<Self> {
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
                .context("vk device creation failed")?
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
                log::warn!("error waiting for device to get idle before destroying: {e}")
            };
            self.device.destroy_device(None);
        }
    }
}

#[derive(Getters, MutGetters)]
pub struct Device {
    pub(crate) dropper: Arc<DeviceDropper>,
    allocator: Arc<Mutex<Allocator>>,
    #[getset(get = "pub", get_mut = "pub")]
    canvas: Canvas,
    cmd_pool: CmdPool,
    sync_pool: SemPool,
}

impl Device {
    pub(crate) fn new(
        instance_dropper: &Arc<InstanceDropper>,
        gpu_id: usize,
    ) -> anyhow::Result<Self> {
        let dropper = DeviceDropper::new(instance_dropper, gpu_id).map(Arc::new)?;
        let sync_pool = SemPool::new(&dropper);
        let cmd_pool = CmdPool::new(&dropper)?;
        let canvas = Canvas::new(&dropper)?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: dropper.instance_dropper.instance.clone(),
            device: dropper.device.clone(),
            physical_device: dropper.gpu_info.handle,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map(Mutex::new)
        .map(Arc::new)?;
        Ok(Self {
            dropper: dropper.clone(),
            allocator,
            canvas,
            cmd_pool,
            sync_pool,
        })
    }

    pub fn new_buffer(&self, info: BufferCreateInfo) -> anyhow::Result<BufferRef> {
        BufferRef::new(&self.dropper, &self.allocator, info)
    }

    pub fn new_image(&self, info: ImageCreateInfo) -> anyhow::Result<ImageRef> {
        ImageRef::new(&self.dropper, &self.allocator, info)
    }

    pub fn new_sampler(&self) -> anyhow::Result<Sampler> {
        Sampler::new(&self.dropper)
    }

    pub fn new_graphics_pipeline(
        &self,
        info: GraphicsPipelineCreateInfo,
    ) -> anyhow::Result<GraphicsPipeline> {
        GraphicsPipeline::new(&self.dropper, info).context("graphics pipeline creation failed")
    }

    pub fn new_task(&self) -> anyhow::Result<Task> {
        let cb = self.cmd_pool.get_cb()?;
        let sem = self.sync_pool.get_sem()?;
        Ok(Task {
            cb,
            sem,
            preserve_bufs: vec![],
            preserve_imgs: vec![],
            preserve_views: vec![],
            preserve_gps: vec![],
            swapchain_image: None,
        })
    }
}
