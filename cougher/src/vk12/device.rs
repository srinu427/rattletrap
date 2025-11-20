use ash::{khr, vk};

use crate::vk12::{
    instance::{Vk12Gpu, Vk12Instance, Vk12InstanceError},
    sync::{reset_fences, wait_for_fences},
};

pub struct SwapchainData {
    pub(crate) images: Vec<vk::Image>,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) present_mode: vk::PresentModeKHR,
    pub(crate) extent: vk::Extent2D,
    pub(crate) surface_fmt: vk::SurfaceFormatKHR,
}

#[derive(Debug, thiserror::Error)]
pub enum Vk12DeviceError {
    #[error("Instance related error: {0}")]
    Vk12InstanceError(#[from] Vk12InstanceError),
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
    #[error("Error waiting for Vulkan Fence: {0}")]
    FenceWaitError(vk::Result),
    #[error("Error reseting for Vulkan Fence: {0}")]
    FenceResetError(vk::Result),
}

pub struct Vk12Device {
    pub(crate) swapchain_data: SwapchainData,
    pub(crate) g_queue_fam: u32,
    pub(crate) g_queue: vk::Queue,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) swapchain_device: khr::swapchain::Device,
    pub(crate) device: ash::Device,
    pub(crate) instance: Vk12Instance,
}

impl Vk12Device {
    fn init_device(instance: &Vk12Instance, gpu: &Vk12Gpu) -> Result<ash::Device, Vk12DeviceError> {
        let queue_priorities = [0.0];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gpu.g_queue_family.0 as _)
            .queue_priorities(&queue_priorities)];
        let extensions = [
            khr::swapchain::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            khr::portability_subset::NAME.as_ptr(),
        ];
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions);
        let device = unsafe {
            instance
                .instance
                .create_device(gpu.physical_device, &device_create_info, None)
                .map_err(Vk12DeviceError::DeviceCreateError)?
        };
        Ok(device)
    }
    fn init_swapchain(
        instance: &Vk12Instance,
        swapchain_device: &khr::swapchain::Device,
        gpu: &Vk12Gpu,
    ) -> Result<SwapchainData, Vk12DeviceError> {
        let (formats, caps, present_modes) = unsafe {
            let formats: Vec<_> = instance
                .surface_instance
                .get_physical_device_surface_formats(gpu.physical_device, instance.surface)
                .map_err(Vk12DeviceError::GetSurfaceFormatsError)?
                .into_iter()
                .filter(|format| {
                    let supported = instance
                        .instance
                        .get_physical_device_format_properties(gpu.physical_device, format.format)
                        .optimal_tiling_features
                        .contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_DST,
                        );
                    supported
                })
                .collect();

            let caps = instance
                .surface_instance
                .get_physical_device_surface_capabilities(gpu.physical_device, instance.surface)
                .map_err(Vk12DeviceError::GetSurfaceCapabilitiesError)?;

            let present_modes = instance
                .surface_instance
                .get_physical_device_surface_present_modes(gpu.physical_device, instance.surface)
                .map_err(Vk12DeviceError::GetPresentModesError)?;
            (formats, caps, present_modes)
        };

        let format = formats
            .iter()
            .filter(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .filter(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    || format.format == vk::Format::B8G8R8A8_SRGB
                    || format.format == vk::Format::R8G8B8A8_UNORM
                    || format.format == vk::Format::R8G8B8A8_SRGB
            })
            .next()
            .cloned()
            .ok_or(Vk12DeviceError::NoSuitableSurfaceFormat)?;

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = instance.window.inner_size();
            extent.width = window_res.width;
            extent.height = window_res.height;
        }

        let present_mode = present_modes
            .iter()
            .filter(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .next()
            .cloned()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_image_count = std::cmp::min(
            caps.min_image_count + 1,
            if caps.max_image_count == 0 {
                std::u32::MAX
            } else {
                caps.max_image_count
            },
        );

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(instance.surface)
            .min_image_count(swapchain_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe {
            swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(Vk12DeviceError::SwapchainCreateError)?
        };

        let swapchain_images = unsafe {
            match swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(Vk12DeviceError::SwapchainGetImagesError)
            {
                Ok(imgs) => imgs,
                Err(e) => {
                    swapchain_device.destroy_swapchain(swapchain, None);
                    return Err(e);
                }
            }
        };
        Ok(SwapchainData {
            images: swapchain_images,
            swapchain,
            present_mode,
            extent,
            surface_fmt: format,
        })
    }

    pub fn new(
        instance: Vk12Instance,
        gpu: Vk12Gpu,
    ) -> Result<Self, (Vk12Instance, Vk12DeviceError)> {
        let device = match Self::init_device(&instance, &gpu) {
            Ok(d) => d,
            Err(e) => return Err((instance, e)),
        };

        let swapchain_device = khr::swapchain::Device::new(&instance.instance, &device);
        let swapchain_data = match Self::init_swapchain(&instance, &swapchain_device, &gpu) {
            Ok(s) => s,
            Err(e) => {
                unsafe {
                    device.destroy_device(None);
                }
                return Err((instance, e));
            }
        };

        let g_queue = unsafe { device.get_device_queue(gpu.g_queue_family.0 as _, 0) };

        Ok(Self {
            swapchain_data,
            g_queue_fam: gpu.g_queue_family.0 as _,
            g_queue,
            physical_device: gpu.physical_device,
            swapchain_device,
            device,
            instance,
        })
    }

    pub fn refresh_swapchain_res(&mut self) -> Result<(), Vk12DeviceError> {
        let caps = unsafe {
            self.instance
                .surface_instance
                .get_physical_device_surface_capabilities(
                    self.physical_device,
                    self.instance.surface,
                )
                .map_err(Vk12DeviceError::GetSurfaceCapabilitiesError)?
        };
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = self.instance.window.inner_size();
            extent.width = window_res.width;
            extent.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.instance.surface)
            .min_image_count(self.swapchain_data.images.len() as _)
            .image_format(self.swapchain_data.surface_fmt.format)
            .image_color_space(self.swapchain_data.surface_fmt.color_space)
            .image_extent(self.swapchain_data.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.swapchain_data.present_mode)
            .old_swapchain(self.swapchain_data.swapchain)
            .clipped(true);

        let new_swapchain = unsafe {
            self.swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(Vk12DeviceError::SwapchainCreateError)?
        };
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain_data.swapchain, None);
        }
        let swapchain_images = unsafe {
            match self
                .swapchain_device
                .get_swapchain_images(new_swapchain)
                .map_err(Vk12DeviceError::SwapchainGetImagesError)
            {
                Ok(imgs) => imgs,
                Err(e) => {
                    self.swapchain_device.destroy_swapchain(new_swapchain, None);
                    return Err(e);
                }
            }
        };

        self.swapchain_data.extent = extent;
        self.swapchain_data.swapchain = new_swapchain;
        self.swapchain_data.images = swapchain_images;
        Ok(())
    }

    pub fn acquire_next_ws_img(
        &mut self,
        fence: vk::Fence,
    ) -> Result<(u32, bool), Vk12DeviceError> {
        let mut refreshed = false;
        loop {
            let aquire_out = unsafe {
                self.swapchain_device.acquire_next_image(
                    self.swapchain_data.swapchain,
                    u64::MAX,
                    vk::Semaphore::null(),
                    fence,
                )
            };

            let (idx, is_suboptimal) = match aquire_out {
                Ok((i, s)) => (Some(i), s),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (None, true),
                Err(e) => return Err(Vk12DeviceError::AcquireNextImageError(e)),
            };

            if is_suboptimal {
                self.refresh_swapchain_res()?;
                refreshed = true;
                if idx.is_some() {
                    wait_for_fences(&self.device, &[fence], None)
                        .map_err(Vk12DeviceError::FenceWaitError)?;
                    reset_fences(&self.device, &[fence])
                        .map_err(Vk12DeviceError::FenceResetError)?;
                }
                continue;
            }
            if let Some(img_idx) = idx {
                wait_for_fences(&self.device, &[fence], None)
                    .map_err(Vk12DeviceError::FenceWaitError)?;
                reset_fences(&self.device, &[fence]).map_err(Vk12DeviceError::FenceResetError)?;
                return Ok((img_idx, refreshed));
            }
        }
    }
}

impl Drop for Vk12Device {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain_data.swapchain, None);
            self.device.destroy_device(None);
        }
    }
}
