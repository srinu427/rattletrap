use std::sync::Arc;

use ash::{ext, khr, vk};
use gpu_allocator::{
    AllocationError,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use thiserror::Error;

use crate::wrappers::instance::Instance;

#[derive(Debug, Clone, Copy)]
pub enum QueueType {
    Graphics,
}

pub fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        ext::descriptor_indexing::NAME.as_ptr(),
        khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

#[derive(Debug, Error)]
pub enum LogicalDeviceError {
    #[error("Vulkan GPU listing error: {0}")]
    ListDevicesError(vk::Result),
    #[error("No suitable GPU found")]
    NoSuitableGpu,
    #[error("Vulkan logical device creation error: {0}")]
    DeviceCreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct LogicalDevice {
    #[get = "pub"]
    swapchain_device: khr::swapchain::Device,
    #[get_copy = "pub"]
    graphics_queue: vk::Queue,
    #[get_copy = "pub"]
    graphics_qf_id: u32,
    #[get = "pub"]
    device: ash::Device,
    #[get_copy = "pub"]
    gpu: vk::PhysicalDevice,
    #[get = "pub"]
    instance: Arc<Instance>,
}

impl LogicalDevice {
    pub fn new(instance: Arc<Instance>) -> Result<Self, LogicalDeviceError> {
        let gpus = unsafe {
            instance
                .instance()
                .enumerate_physical_devices()
                .map_err(LogicalDeviceError::ListDevicesError)?
        };

        let mut gpu_w_qf_ids = gpus
            .iter()
            .filter_map(|&gpu| select_graphics_queue(&instance, gpu).map(|qf_id| (gpu, qf_id)))
            .collect::<Vec<_>>();

        gpu_w_qf_ids.sort_by_key(|(gpu, _)| gpu_weight(&instance, *gpu));

        let (gpu, graphics_qf_id) = gpu_w_qf_ids
            .pop()
            .ok_or(LogicalDeviceError::NoSuitableGpu)?;

        let queue_priorities = [1.0];
        let queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_qf_id)
                .queue_priorities(&queue_priorities),
        ];

        let device_extensions = get_device_extensions();
        let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true);
        let mut dynamic_rendering_switch =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features)
            .push_next(&mut device_12_features)
            .push_next(&mut dynamic_rendering_switch);

        let device = unsafe {
            instance
                .instance()
                .create_device(gpu, &device_create_info, None)
                .map_err(LogicalDeviceError::DeviceCreateError)?
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_qf_id, 0) };

        let swapchain_device = khr::swapchain::Device::new(&instance.instance(), &device);

        Ok(Self {
            swapchain_device,
            graphics_queue,
            graphics_qf_id,
            device,
            gpu,
            instance,
        })
    }

    pub fn make_allocator(&self) -> Result<Allocator, AllocationError> {
        Allocator::new(&AllocatorCreateDesc {
            instance: self.instance().instance().clone(),
            device: self.device().clone(),
            physical_device: self.gpu(),
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
    }
}

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

fn select_graphics_queue(instance: &Instance, gpu: vk::PhysicalDevice) -> Option<u32> {
    let queue_families = unsafe {
        instance
            .instance()
            .get_physical_device_queue_family_properties(gpu)
    };
    queue_families
        .iter()
        .enumerate()
        .filter(|(i, queue_family)| {
            let supports_graphics = queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_present = unsafe {
                instance
                    .surface_instance()
                    .get_physical_device_surface_support(gpu, *i as u32, instance.surface())
                    .unwrap_or(false)
            };
            supports_graphics && supports_present
        })
        .max_by_key(|(_, queue_family)| queue_family.queue_count)
        .map(|(i, _)| i as u32)
}

fn gpu_weight(instance: &Instance, gpu: vk::PhysicalDevice) -> u32 {
    let properties = unsafe { instance.instance().get_physical_device_properties(gpu) };

    let mut weight = 0;

    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        weight += 1;
    }
    weight
}
