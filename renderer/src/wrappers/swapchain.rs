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
    #[error("Vulkan swapchain images listing error: {0}")]
    GetImagesError(vk::Result),
    #[error("Vulkan image view creation error: {0}")]
    CreateImageViewError(vk::Result),
}

fn fetch_images_make_views(
    device: &LogicalDevice,
    swapchain: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
) -> Result<(Vec<vk::Image>, Vec<vk::ImageView>), SwapchainError> {
    let images = unsafe {
        device
            .swapchain_device()
            .get_swapchain_images(swapchain)
            .map_err(SwapchainError::GetImagesError)?
    };

    let mut image_views = Vec::with_capacity(images.len());
    for &image in &images {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe {
            device
                .device()
                .create_image_view(&create_info, None)
                .map_err(SwapchainError::CreateImageViewError)?
        };
        image_views.push(image_view);
    }

    Ok((images, image_views))
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Swapchain {
    #[get = "pub"]
    image_views: Vec<vk::ImageView>,
    #[get = "pub"]
    images: Vec<vk::Image>,
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
        let surface_instance = device.instance().surface_instance();
        let surface = device.instance().surface();
        let instance = device.instance().instance();

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

        let (images, image_views) = fetch_images_make_views(&device, swapchain, format)?;

        Ok(Self {
            image_views,
            images,
            swapchain,
            format,
            extent,
            device,
        })
    }

    pub fn refresh_resolution(&mut self) -> Result<(), SwapchainError> {
        let surface_instance = self.device.instance().surface_instance();
        let surface = self.device.instance().surface();

        let caps = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(self.device.gpu(), surface)
                .map_err(SwapchainError::GetSurfaceCapabilitiesError)?
        };

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = self.device.instance().window().inner_size();
            extent.width = window_res.width;
            extent.height = window_res.height;
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device.instance().surface())
            .min_image_count(self.images.len() as u32)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.swapchain);

        let swapchain = unsafe {
            self.device
                .swapchain_device()
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainError::SwapchainCreateError)?
        };

        let (images, image_views) = fetch_images_make_views(&self.device, swapchain, self.format)?;

        unsafe {
            for &image_view in &self.image_views {
                self.device.device().destroy_image_view(image_view, None);
            }
            self.device
                .swapchain_device()
                .destroy_swapchain(self.swapchain, None);
        }

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;
        self.extent = extent;
        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for &image_view in &self.image_views {
                self.device.device().destroy_image_view(image_view, None);
            }
            self.device
                .swapchain_device()
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
