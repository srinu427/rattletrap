use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::vk_wrap::{
    device::Device,
    image_2d::Image2d,
    sync::{Fence, SemStageInfo, SyncError},
};

#[derive(Debug, thiserror::Error)]
pub enum SwapchainError {
    #[error("error getting Vulkan Surface Formats: {0}")]
    GetSurfaceFormatsError(vk::Result),
    #[error("error getting Vulkan Surface Capabilities: {0}")]
    GetSurfaceCapabilitiesError(vk::Result),
    #[error("error getting Vulkan Present Modes: {0}")]
    GetPresentModesError(vk::Result),
    #[error("No suitable surface format found")]
    NoSuitableSurfaceFormat,
    #[error("error creating Vulkan Swapchain: {0}")]
    SwapchainCreateError(vk::Result),
    #[error("error getting Vulkan Swapchain Images: {0}")]
    SwapchainGetImagesError(vk::Result),
    #[error("error acquiring next Vulkan Swapchain Image: {0}")]
    AcquireNextImageError(vk::Result),
    #[error("error presenting Vulkan Swapchain Image: {0}")]
    PresentError(vk::Result),
    #[error("Fence related error: {0}")]
    FenceError(#[from] SyncError),
}

pub struct Swapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) extent: vk::Extent2D,
    pub(crate) images: Vec<Image2d>,
    pub(crate) surface_fmt: vk::SurfaceFormatKHR,
    pub(crate) present_mode: vk::PresentModeKHR,
    pub(crate) device: Arc<Device>,
}

impl Swapchain {
    pub fn new(device: &Arc<Device>) -> Result<Swapchain, SwapchainError> {
        let (formats, caps, present_modes) = unsafe {
            let formats: Vec<_> = device
                .instance
                .surface_instance
                .get_physical_device_surface_formats(
                    device.physical_device,
                    device.instance.surface,
                )
                .map_err(SwapchainError::GetSurfaceFormatsError)?
                .into_iter()
                .filter(|format| {
                    let supported = device
                        .instance
                        .instance
                        .get_physical_device_format_properties(
                            device.physical_device,
                            format.format,
                        )
                        .optimal_tiling_features
                        .contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_DST,
                        );
                    supported
                })
                .collect();

            let caps = device
                .instance
                .surface_instance
                .get_physical_device_surface_capabilities(
                    device.physical_device,
                    device.instance.surface,
                )
                .map_err(SwapchainError::GetSurfaceCapabilitiesError)?;

            let present_modes = device
                .instance
                .surface_instance
                .get_physical_device_surface_present_modes(
                    device.physical_device,
                    device.instance.surface,
                )
                .map_err(SwapchainError::GetPresentModesError)?;
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
            .ok_or(SwapchainError::NoSuitableSurfaceFormat)?;

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = device.instance.window.inner_size();
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
            .surface(device.instance.surface)
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
            device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainError::SwapchainCreateError)?
        };

        let swapchain_images = unsafe {
            match device
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(SwapchainError::SwapchainGetImagesError)
            {
                Ok(imgs) => imgs,
                Err(e) => {
                    device.swapchain_device.destroy_swapchain(swapchain, None);
                    return Err(e);
                }
            }
        };

        let images = swapchain_images
            .into_iter()
            .map(|i| Image2d {
                image: i,
                memory: None,
                view: vk::ImageView::null(),
                location: MemoryLocation::GpuOnly,
                extent,
                format: format.format,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                allocator: None,
                device: device.clone(),
            })
            .collect();

        Ok(Self {
            swapchain,
            extent,
            images,
            surface_fmt: format,
            present_mode,
            device: device.clone(),
        })
    }

    pub fn refresh_swapchain_res(&mut self) -> Result<(), SwapchainError> {
        let caps = unsafe {
            self.device
                .instance
                .surface_instance
                .get_physical_device_surface_capabilities(
                    self.device.physical_device,
                    self.device.instance.surface,
                )
                .map_err(SwapchainError::GetSurfaceCapabilitiesError)?
        };
        let mut extent = caps.current_extent;
        // println!("{:#?}", caps);
        if extent.width == u32::MAX || extent.height == u32::MAX {
            // println!(
            //     "invalid current extent: {:?}. using windows resolution",
            //     extent
            // );
            let window_res = self.device.instance.window.inner_size();
            extent.width = window_res.width;
            extent.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device.instance.surface)
            .min_image_count(self.images.len() as _)
            .image_format(self.surface_fmt.format)
            .image_color_space(self.surface_fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .old_swapchain(self.swapchain)
            .clipped(true);

        let new_swapchain = unsafe {
            self.device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainError::SwapchainCreateError)?
        };
        unsafe {
            self.device
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
        let swapchain_images = unsafe {
            match self
                .device
                .swapchain_device
                .get_swapchain_images(new_swapchain)
                .map_err(SwapchainError::SwapchainGetImagesError)
            {
                Ok(imgs) => imgs,
                Err(e) => {
                    self.device
                        .swapchain_device
                        .destroy_swapchain(new_swapchain, None);
                    return Err(e);
                }
            }
        };
        let images = swapchain_images
            .into_iter()
            .map(|i| Image2d {
                image: i,
                memory: None,
                view: vk::ImageView::null(),
                location: MemoryLocation::GpuOnly,
                extent,
                format: self.surface_fmt.format,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                allocator: None,
                device: self.device.clone(),
            })
            .collect();

        self.extent = extent;
        self.swapchain = new_swapchain;
        self.images = images;
        Ok(())
    }

    pub fn acquire_next_img(&mut self, fence: &Fence) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.device.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                fence.fence,
            )
        }
    }

    pub fn present_image(
        &self,
        idx: u32,
        wait_sems: &[SemStageInfo],
    ) -> Result<(), SwapchainError> {
        let wait_sems_vk: Vec<_> = wait_sems.iter().map(|s| s.sem.sem).collect();
        unsafe {
            self.device
                .swapchain_device
                .queue_present(
                    self.device.g_queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.swapchain])
                        .image_indices(&[idx])
                        .wait_semaphores(&wait_sems_vk),
                )
                .map_err(SwapchainError::PresentError)?;
        }
        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
