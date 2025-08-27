use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(Debug, Error)]
pub enum SwapchainError {
    #[error("Vulkan surface format listing error: {0}")]
    GetSurfaceFormatsError(vk::Result),
    #[error("Vulkan surface capabilities listing error: {0}")]
    GetSurfaceCapabilitiesError(vk::Result),
    #[error("Vulkan present mode listing error: {0}")]
    GetPresentModesError(vk::Result),
    #[error("No suitable surface format found")]
    NoSuitableSurfaceFormat,
    #[error("Vulkan swapchain creation error: {0}")]
    SwapchainCreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Swapchain {
    #[get_copy = "pub"]
    swapchain: vk::SwapchainKHR,
    #[get_copy = "pub"]
    format: vk::SurfaceFormatKHR,
    #[get_copy = "pub"]
    extent: vk::Extent2D,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Swapchain {
    pub fn new(device: Arc<LogicalDevice>) -> Result<Self, SwapchainError> {
        let surface_instance = &device.instance().surface_instance();
        let surface = device.instance().surface();
        let instance = &device.instance().instance();

        let formats = unsafe {
            surface_instance
                .get_physical_device_surface_formats(device.gpu(), surface)
                .map_err(SwapchainError::GetSurfaceFormatsError)?
        };

        let caps = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(device.gpu(), surface)
                .map_err(SwapchainError::GetSurfaceCapabilitiesError)?
        };

        let present_modes = unsafe {
            surface_instance
                .get_physical_device_surface_present_modes(device.gpu(), surface)
                .map_err(SwapchainError::GetPresentModesError)?
        };

        let format = formats
            .iter()
            .filter(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .filter(|format| {
                let supported = unsafe {
                    instance
                        .get_physical_device_format_properties(device.gpu(), format.format)
                        .optimal_tiling_features
                        .contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_DST
                                | vk::FormatFeatureFlags::STORAGE_IMAGE,
                        )
                };
                supported
                    && (format.format == vk::Format::B8G8R8A8_UNORM
                        || format.format == vk::Format::R8G8B8A8_UNORM
                        || format.format == vk::Format::B8G8R8A8_SRGB
                        || format.format == vk::Format::R8G8B8A8_SRGB)
            })
            .next()
            .cloned()
            .ok_or(SwapchainError::NoSuitableSurfaceFormat)?;

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = device.instance().window().inner_size();
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
            .surface(surface)
            .min_image_count(swapchain_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe {
            device
                .swapchain_device()
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainError::SwapchainCreateError)?
        };

        Ok(Self {
            swapchain,
            format,
            extent,
            device,
        })
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device
                .swapchain_device()
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
