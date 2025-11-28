use ash::{khr, vk};
use gpu_allocator::AllocationError;

use crate::vk_wrap::instance::{Gpu, Instance, InstanceError};

#[derive(Debug, thiserror::Error)]
pub enum AllocError {
    #[error("Error allocating GPU memory: {0}")]
    LibError(#[from] AllocationError),
    #[error("Error acquiring GPU Allocator Mutex Lock: {0}")]
    LockError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Instance related error: {0}")]
    Vk12InstanceError(#[from] InstanceError),
    #[error("Error creating Vulkan Device: {0}")]
    DeviceCreateError(vk::Result),
    #[error("Error getting surface formats: {0}")]
    GetSurfaceFormatsError(vk::Result),
    #[error("Error getting surface capabilities: {0}")]
    GetSurfaceCapabilitiesError(vk::Result),
    #[error("Error getting present modes: {0}")]
    GetPresentModesError(vk::Result),
    #[error("No supported surface formats found")]
    NoSuitableSurfaceFormat,
    #[error("Error creating Vulkan Swapchain: {0}")]
    SwapchainCreateError(vk::Result),
    #[error("Error getting Vulkan Swapchain Images: {0}")]
    SwapchainGetImagesError(vk::Result),
    #[error("Error acquiring next Vulkan Swapchain Image to present: {0}")]
    AcquireNextImageError(vk::Result),
}

pub struct Device {
    pub(crate) g_queue_fam: u32,
    pub(crate) g_queue: vk::Queue,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) swapchain_device: khr::swapchain::Device,
    pub(crate) device: ash::Device,
    pub(crate) instance: Instance,
}

impl Device {
    fn init_device(instance: &Instance, gpu: &Gpu) -> Result<ash::Device, DeviceError> {
        let queue_priorities = [0.0];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gpu.g_queue_family.0 as _)
            .queue_priorities(&queue_priorities)];
        let extensions = [
            khr::swapchain::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            khr::portability_subset::NAME.as_ptr(),
        ];
        let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true);
        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&device_features)
            .push_next(&mut device_12_features);
        let device = unsafe {
            instance
                .instance
                .create_device(gpu.physical_device, &device_create_info, None)
                .map_err(DeviceError::DeviceCreateError)?
        };
        Ok(device)
    }

    pub fn new(instance: Instance, gpu: Gpu) -> Result<Self, (Instance, DeviceError)> {
        let device = match Self::init_device(&instance, &gpu) {
            Ok(d) => d,
            Err(e) => return Err((instance, e)),
        };

        let swapchain_device = khr::swapchain::Device::new(&instance.instance, &device);

        let g_queue = unsafe { device.get_device_queue(gpu.g_queue_family.0 as _, 0) };

        Ok(Self {
            g_queue_fam: gpu.g_queue_family.0 as _,
            g_queue,
            physical_device: gpu.physical_device,
            swapchain_device,
            device,
            instance,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
