use std::sync::{Arc, Mutex};

use anyhow::Context;
#[cfg(debug_assertions)]
use ash::ext;
use ash::khr;
#[cfg(target_os = "macos")]
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::vkraii::command::{CommandBufferRaii, CommandPoolRaii, Task};

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
            .with_context(|| "instance creation failed")?
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
                .with_context(|| "get display handle failed")?
                .as_raw(),
            window
                .window_handle()
                .with_context(|| "get window handle failed")?
                .as_raw(),
            None,
        )
        .with_context(|| "surface creation failed")?
    };
    Ok(surface)
}

pub struct InstanceDropper {
    pub surface: vk::SurfaceKHR,
    pub surface_instance: khr::surface::Instance,
    pub instance: ash::Instance,
    pub window: Arc<Window>,
    _entry: ash::Entry,
}

impl InstanceDropper {
    fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_instance(&entry)?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, window)?;
        Ok(Self {
            surface,
            surface_instance,
            instance,
            window: window.clone(),
            _entry: entry,
        })
    }
}

impl Drop for InstanceDropper {
    fn drop(&mut self) {
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct InstanceRaii {
    instance_d: Arc<InstanceDropper>,
}

impl InstanceRaii {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        Ok(Self {
            instance_d: Arc::new(InstanceDropper::new(window)?),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SelectedGpuInfo {
    pub gpu: vk::PhysicalDevice,
    pub graphics_qf: u32,
}

pub fn select_gpu(
    instance: &ash::Instance,
    surface_instance: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<SelectedGpuInfo> {
    let gpus = unsafe { instance.enumerate_physical_devices()? };
    let mut supported_gpus = vec![];
    for gpu in &gpus {
        let qf_props = unsafe { instance.get_physical_device_queue_family_properties(*gpu) };
        let graphics_qf = qf_props
            .into_iter()
            .enumerate()
            .filter(|(_idx, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .filter(|(idx, _props)| unsafe {
                surface_instance
                    .get_physical_device_surface_support(*gpu, *idx as _, surface)
                    .unwrap_or(false)
            })
            .max_by_key(|(_idx, props)| props.queue_count);
        if let Some((qf, qf_prop)) = graphics_qf {
            let gpu_props = unsafe { instance.get_physical_device_properties(*gpu) };
            let mem_props = unsafe { instance.get_physical_device_memory_properties(*gpu) };
            supported_gpus.push((gpu, qf, qf_prop, gpu_props, mem_props));
        }
    }
    supported_gpus.sort_by_key(|g| g.4.memory_heaps.iter().map(|h| h.size).sum::<u64>());
    let selected_gpu = supported_gpus
        .iter()
        .find(|g| g.3.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        .unwrap_or(supported_gpus.first().context("no supported gpus found")?);
    Ok(SelectedGpuInfo {
        gpu: *selected_gpu.0,
        graphics_qf: selected_gpu.1 as _,
    })
}

fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        // ext::descriptor_indexing::NAME.as_ptr(),
        // khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

pub fn create_device(
    instance: &ash::Instance,
    selected_gpu: &SelectedGpuInfo,
) -> anyhow::Result<(ash::Device, vk::Queue)> {
    let queue_priorities = [1.0];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected_gpu.graphics_qf)
        .queue_priorities(&queue_priorities)];
    let device_extensions = get_device_extensions();
    let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
        // .descriptor_indexing(true)
        // .runtime_descriptor_array(true)
        // .shader_sampled_image_array_non_uniform_indexing(true)
        // .descriptor_binding_sampled_image_update_after_bind(true)
        // .descriptor_binding_partially_bound(true)
        // .descriptor_binding_variable_descriptor_count(true)
        .timeline_semaphore(true);
    let device_features = vk::PhysicalDeviceFeatures::default();
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&device_features)
        .push_next(&mut device_12_features);
    let device = unsafe {
        instance
            .create_device(selected_gpu.gpu, &device_create_info, None)
            .context("vk device creation failed")?
    };
    let gfx_queue = unsafe { device.get_device_queue(selected_gpu.graphics_qf, 0) };
    Ok((device, gfx_queue))
}

pub struct DeviceDropper {
    pub graphics_queue: vk::Queue,
    pub graphics_qf: u32,
    pub swapchain_device: khr::swapchain::Device,
    pub device: ash::Device,
    pub gpu: vk::PhysicalDevice,
    pub instance_raii: Arc<InstanceDropper>,
}

impl DeviceDropper {
    fn new(instance: &InstanceRaii) -> anyhow::Result<Self> {
        let selected_gpu = select_gpu(
            &instance.instance_d.instance,
            &instance.instance_d.surface_instance,
            instance.instance_d.surface,
        )?;
        let (device, queue) = create_device(&instance.instance_d.instance, &selected_gpu)?;
        let swapchain_device = khr::swapchain::Device::new(&instance.instance_d.instance, &device);
        Ok(Self {
            graphics_queue: queue,
            graphics_qf: selected_gpu.graphics_qf,
            swapchain_device,
            device,
            gpu: selected_gpu.gpu,
            instance_raii: instance.instance_d.clone(),
        })
    }
}

impl Drop for DeviceDropper {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub struct DeviceRaii {
    pub semaphore: vk::Semaphore,
    pub current_sub_id: u64,
    pub command_pool: CommandPoolRaii,
    pub allocator: Arc<Mutex<Allocator>>,
    pub device_d: Arc<DeviceDropper>,
}

impl DeviceRaii {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let instance = InstanceRaii::new(window)?;
        let device_d = Arc::new(DeviceDropper::new(&instance)?);
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.instance_d.instance.clone(),
            device: device_d.device.clone(),
            physical_device: device_d.gpu,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        let command_pool = CommandPoolRaii::new(&device_d)?;
        let semaphore = unsafe {
            device_d.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(
                    &mut vk::SemaphoreTypeCreateInfo::default()
                        .semaphore_type(vk::SemaphoreType::TIMELINE),
                ),
                None,
            )?
        };
        Ok(Self {
            semaphore,
            current_sub_id: 0,
            command_pool,
            allocator: Arc::new(Mutex::new(allocator)),
            device_d,
        })
    }

    pub fn run_commands(
        &mut self,
        mut command_buffers: Vec<CommandBufferRaii>,
    ) -> anyhow::Result<Task> {
        self.current_sub_id += 1;
        for cb in &mut command_buffers {
            cb.end()?;
        }
        unsafe {
            self.device_d.device.queue_submit(
                self.device_d.graphics_queue,
                &[vk::SubmitInfo::default()
                    .command_buffers(
                        &command_buffers
                            .iter()
                            .map(|cb| cb.command_buffer)
                            .collect::<Vec<_>>(),
                    )
                    .signal_semaphores(&[self.semaphore])
                    .push_next(
                        &mut vk::TimelineSemaphoreSubmitInfo::default()
                            .signal_semaphore_values(&[self.current_sub_id]),
                    )],
                vk::Fence::null(),
            )?;
        }

        Ok(Task {
            command_buffers,
            task_val: self.current_sub_id,
        })
    }

    pub fn wait_on_task(&mut self, task: Task) -> anyhow::Result<()> {
        unsafe {
            self.device_d.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[self.semaphore])
                    .values(&[task.task_val]),
                u64::MAX,
            )?;
        }
        Ok(())
    }
}

impl Drop for DeviceRaii {
    fn drop(&mut self) {
        unsafe {
            let _ = self
                .device_d
                .device
                .device_wait_idle()
                .inspect_err(|e| log::warn!("waiting for device to complete work failed: {e}"))
                .ok();
            self.device_d.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
