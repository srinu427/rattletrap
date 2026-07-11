use anyhow::Context;
#[cfg(debug_assertions)]
use ash::ext;
use ash::{khr, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

pub struct ImageAccess {
    pub layout: vk::ImageLayout,
    pub access_flags: vk::AccessFlags,
    pub access_stage: vk::PipelineStageFlags,
}

pub struct StagingBuffer {
    pub buffer: vk::Buffer,
    pub mem: Allocation,
    pub size: u64,
}

impl StagingBuffer {
    pub fn new(device: &ash::Device, allocator: &mut Allocator, size: u64) -> anyhow::Result<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let (buffer, mem) =
            create_buffer(device, allocator, &create_info, MemoryLocation::CpuToGpu)?;
        Ok(Self { buffer, mem, size })
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        allocator
            .free(self.mem)
            .inspect_err(|e| log::warn!("freeing memory of buffer {:?} failed {e}", self.buffer))
            .ok();
    }
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

pub fn create_command_pool(
    device: &ash::Device,
    queue_family: u32,
) -> anyhow::Result<vk::CommandPool> {
    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family),
            None,
        )?
    };
    Ok(command_pool)
}

pub fn allocate_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: usize,
) -> anyhow::Result<Vec<vk::CommandBuffer>> {
    let command_buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(count as _)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY),
        )?
    };
    Ok(command_buffers)
}

pub fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    create_info: &vk::BufferCreateInfo,
    mem_location: MemoryLocation,
) -> anyhow::Result<(vk::Buffer, Allocation)> {
    let buffer = unsafe { device.create_buffer(&create_info, None)? };
    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem = allocator.allocate(&AllocationCreateDesc {
        name: &format!("{:?}", buffer),
        requirements: mem_req,
        location: mem_location,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    unsafe {
        device.bind_buffer_memory(buffer, mem.memory(), mem.offset())?;
    }
    Ok((buffer, mem))
}

pub fn create_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    create_info: &vk::ImageCreateInfo,
) -> anyhow::Result<(vk::Image, Allocation)> {
    let image = unsafe { device.create_image(&create_info, None)? };
    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let mem = allocator.allocate(&AllocationCreateDesc {
        name: &format!("{:?}", image),
        requirements: mem_req,
        location: MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    unsafe {
        device.bind_image_memory(image, mem.memory(), mem.offset())?;
    }
    Ok((image, mem))
}
