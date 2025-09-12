use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::wrappers::{
    command::{BarrierCommand, Command}, fence::{Fence, FenceError}, image::{Image, ImageAccess}, image_view::{ImageView, ImageViewError}, logical_device::LogicalDevice, semaphore::{Semaphore, SemaphoreError}
};

#[derive(Debug, Error)]
pub enum SwapchainError {
    #[error("Vulkan semaphore error: {0}")]
    SemaphoreError(#[from] SemaphoreError),
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
    #[error("Vulkan image view error: {0}")]
    ImageViewError(#[from] ImageViewError),
    #[error("Vulkan acquire next image error: {0}")]
    AcquireNextImageError(vk::Result),
    #[error("Vulkan fence error: {0}")]
    FenceError(#[from] FenceError),
    #[error("Error during swapchain presentation: {0}")]
    PresentError(vk::Result),
}

fn fetch_images_make_views(
    device: Arc<LogicalDevice>,
    swapchain: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
) -> Result<Vec<Arc<ImageView>>, SwapchainError> {
    let images = unsafe {
        device
            .swapchain_device()
            .get_swapchain_images(swapchain)
            .map_err(SwapchainError::GetImagesError)?
    };
    let images = images
        .iter()
        .map(|image| Image::from_swapchain_image(device.clone(), *image, format.format, extent))
        .map(Arc::new)
        .collect::<Vec<_>>();

    let image_views = images
        .iter()
        .map(|img| {
            ImageView::new(
                img.clone(),
                vk::ImageViewType::TYPE_2D,
                img.full_subresource_range(),
            )
            .map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(image_views)
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Swapchain {
    acquire_semaphore: Semaphore,
    fence: Fence,
    #[get = "pub"]
    image_views: Vec<Arc<ImageView>>,
    #[get_copy = "pub"]
    swapchain: vk::SwapchainKHR,
    #[get_copy = "pub"]
    format: vk::SurfaceFormatKHR,
    #[get_copy = "pub"]
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Swapchain {
    pub fn new(device: Arc<LogicalDevice>) -> Result<(Self, Vec<Command>), SwapchainError> {
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
                format.format == vk::Format::B8G8R8A8_UNORM
                    || format.format == vk::Format::R8G8B8A8_UNORM
                    || format.format == vk::Format::B8G8R8A8_SRGB
                    || format.format == vk::Format::R8G8B8A8_SRGB
            })
            .filter(|format| {
                let supported = unsafe {
                    instance
                        .get_physical_device_format_properties(device.gpu(), format.format)
                        .optimal_tiling_features
                        .contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_DST, // | vk::FormatFeatureFlags::STORAGE_IMAGE,
                        )
                };
                supported
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

        // let present_mode = vk::PresentModeKHR::FIFO;

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
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST, // | vk::ImageUsageFlags::STORAGE,
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

        let image_views = fetch_images_make_views(device.clone(), swapchain, format, extent)?;

        let acquire_semaphore = Semaphore::new(device.clone())?;
        let fence = Fence::new(device.clone(), false)?;

        let commands = image_views
            .iter()
            .map(|iv| {
                Command::Barrier(BarrierCommand::new_image_2d_barrier(
                    iv.image(),
                    ImageAccess::Undefined,
                    ImageAccess::Present
                ))
            })
            .collect();
        Ok((Self {
            acquire_semaphore,
            fence,
            present_mode,
            image_views,
            swapchain,
            format,
            extent,
            device,
        },
        commands))
    }

    pub fn refresh_resolution(&mut self) -> Result<Vec<Command>, SwapchainError> {
        println!("refreshing sw res");
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

        println!("new_res: {:?}", extent);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device.instance().surface())
            .min_image_count(self.image_views.len() as u32)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST, // | vk::ImageUsageFlags::STORAGE,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain);

        self.image_views.clear();

        let swapchain = unsafe {
            self.device
                .swapchain_device()
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainError::SwapchainCreateError)?
        };

        println!("new swapchain created");

        unsafe {
            self.device
                .swapchain_device()
                .destroy_swapchain(self.swapchain, None);
        }

        let image_views =
            fetch_images_make_views(self.device.clone(), swapchain, self.format, extent)?;

        self.image_views = image_views;

        self.swapchain = swapchain;

        self.extent = extent;
        
        let commands = self
            .image_views
            .iter()
            .map(|iv| {
                Command::Barrier(BarrierCommand::new_image_2d_barrier(
                    iv.image(),
                    ImageAccess::Undefined,
                    ImageAccess::Present
                ))
            })
            .collect();

        Ok(commands)
    }

    pub fn acquire_image(&mut self) -> Result<(u32, Vec<Command>), SwapchainError> {
        let mut commands = vec![];
        loop {
            let aquire_out = unsafe {
                self.device.swapchain_device().acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    vk::Semaphore::null(),
                    self.fence.fence(),
                )
            };

            let (idx, is_suboptimal) = match aquire_out {
                Ok((i, s)) => (Some(i), s),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (None, true),
                Err(e) => return Err(SwapchainError::AcquireNextImageError(e)),
            };

            if is_suboptimal {
                commands = self.refresh_resolution()?;
                if idx.is_some() {
                    self.fence.wait(u64::MAX)?;
                    self.fence.reset()?;
                }
                continue;
            }
            if let Some(img_idx) = idx {
                self.fence.wait(u64::MAX)?;
                self.fence.reset()?;
                return Ok((img_idx, commands));
            }
        }
    }

    pub fn present(
        &self,
        image_index: u32,
        wait_semaphores: &[&Semaphore],
    ) -> Result<(), vk::Result> {
        let wait_semaphores_vk = wait_semaphores
            .iter()
            .map(|s| s.semaphore())
            .collect::<Vec<_>>();

        unsafe {
            self.device
                .swapchain_device()
                .queue_present(
                    self.device.graphics_queue(),
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&wait_semaphores_vk)
                        .swapchains(&[self.swapchain])
                        .image_indices(&[image_index]),
                )
                .inspect_err(|e| eprintln!("error during present: {e}"))?;
        }
        Ok(())
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
