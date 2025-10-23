use std::{mem::ManuallyDrop, sync::Arc, u64};

use ash::{ext, khr, vk};
use enumflags2::BitFlags;
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{AllocationCreateDesc, AllocationScheme},
};
use hashbrown::HashMap;
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};

use crate::{
    backends::vulkan_common::{
        VkMemAllocation, VkMemAllocator, buffer_usage_to_vk, format_to_vk,
        image_2d_subresource_layers, image_2d_subresource_range, res_to_extent_2d,
        res_to_extent_3d, vk_to_format,
    },
    traits::{
        ApiLoader, Buffer, BufferUsage, CommandBuffer, CpuFuture, GpuContext, GpuFuture, GpuInfo,
        Image2d, ImageFormat, ImageUsage, QueueType, Resolution2d, Swapchain,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum V12BufferError {
    #[error("Error creating Vulkan buffer object")]
    CreateError(vk::Result),
    #[error("Error with Mutex locks. Addl Context: {0}")]
    LockError(String),
    #[error("Error allocation memory: {0}")]
    AllocationError(#[from] AllocationError),
    #[error("Error binding memory to buffer: {0}")]
    MemoryBindError(vk::Result),
    #[error("No Memory bound to the buffer")]
    NoBoundMemory,
    #[error("Memory not host accessible")]
    MemoryNotHostAccessible,
}

pub struct V12Buffer {
    name: String,
    buffer: vk::Buffer,
    size: u64,
    usage: BitFlags<BufferUsage>,
    memory: Option<VkMemAllocation>,
    device: Arc<V12Device>,
}

impl V12Buffer {
    pub fn new(
        device: Arc<V12Device>,
        allocator: &mut VkMemAllocator,
        gpu_local: bool,
        name: &str,
        size: u64,
        usage: BitFlags<BufferUsage>,
    ) -> Result<Self, V12BufferError> {
        let usage_flags = buffer_usage_to_vk(usage);
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage_flags);
        let buffer = unsafe {
            device
                .device
                .create_buffer(&buffer_create_info, None)
                .map_err(V12BufferError::CreateError)?
        };

        let reqs = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let location = if gpu_local {
            MemoryLocation::GpuOnly
        } else {
            MemoryLocation::CpuToGpu
        };
        let mem = allocator
            .allocator
            .lock()
            .map_err(|e| V12BufferError::LockError(format!("at locking allocator mutex: {e}")))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements: reqs,
                location: location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe {
            device
                .device
                .bind_buffer_memory(buffer, mem.memory(), mem.offset())
                .map_err(V12BufferError::MemoryBindError)?;
        }
        let memory = Some(VkMemAllocation {
            allocation: ManuallyDrop::new(mem),
            allocator: allocator.allocator.clone(),
            is_gpu_local: gpu_local,
        });

        Ok(Self {
            name: name.to_string(),
            buffer,
            size,
            usage,
            memory,
            device,
        })
    }
}

impl Buffer for V12Buffer {
    type AllocatorType = VkMemAllocator;

    type MemType = VkMemAllocation;

    type E = V12BufferError;

    fn name(&self) -> &str {
        &self.name
    }

    fn write_data(&mut self, offset: u64, data: &[u8]) -> Result<(), Self::E> {
        let offset = offset as usize;
        let mem = self.memory.as_mut().ok_or(V12BufferError::NoBoundMemory)?;
        let slice = mem
            .allocation
            .mapped_slice_mut()
            .ok_or(V12BufferError::MemoryNotHostAccessible)?;
        slice[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn usage(&self) -> BitFlags<BufferUsage> {
        self.usage
    }
}

impl Drop for V12Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
        }
    }
}

pub fn image_usage_to_layout(usage: ImageUsage, format: ImageFormat) -> vk::ImageLayout {
    match usage {
        ImageUsage::None => vk::ImageLayout::UNDEFINED,
        ImageUsage::CopySrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageUsage::CopyDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageUsage::PipelineAttachment => {
            if format.is_depth() {
                if format.has_stencil() {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                }
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            }
        }
        ImageUsage::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

pub fn image_usage_to_access(usage: ImageUsage, format: ImageFormat) -> vk::AccessFlags {
    match usage {
        ImageUsage::None => vk::AccessFlags::empty(),
        ImageUsage::CopySrc => vk::AccessFlags::TRANSFER_READ,
        ImageUsage::CopyDst => vk::AccessFlags::TRANSFER_WRITE,
        ImageUsage::PipelineAttachment => {
            if format.is_depth() {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            }
        }
        ImageUsage::Present => vk::AccessFlags::empty(),
    }
}

pub fn image_usage_to_stage(usage: ImageUsage) -> vk::PipelineStageFlags {
    match usage {
        ImageUsage::None => vk::PipelineStageFlags::ALL_COMMANDS,
        ImageUsage::CopySrc => vk::PipelineStageFlags::TRANSFER,
        ImageUsage::CopyDst => vk::PipelineStageFlags::TRANSFER,
        ImageUsage::PipelineAttachment => vk::PipelineStageFlags::FRAGMENT_SHADER,
        ImageUsage::Present => vk::PipelineStageFlags::ALL_COMMANDS,
    }
}

pub fn image_usage_to_vk(usages: BitFlags<ImageUsage>, format: ImageFormat) -> vk::ImageUsageFlags {
    let mut vk_flags = vk::ImageUsageFlags::default();
    for usage in usages {
        match usage {
            ImageUsage::None => {}
            ImageUsage::CopySrc => vk_flags |= vk::ImageUsageFlags::TRANSFER_SRC,
            ImageUsage::CopyDst => vk_flags |= vk::ImageUsageFlags::TRANSFER_DST,
            ImageUsage::PipelineAttachment => {
                if format.is_depth() {
                    vk_flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk_flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT
                }
            }
            ImageUsage::Present => {}
        }
    }
    vk_flags
}

pub fn image_usage_to_feature(
    usages: BitFlags<ImageUsage>,
    format: ImageFormat,
) -> vk::FormatFeatureFlags {
    let mut vk_flags = vk::FormatFeatureFlags::default();
    for usage in usages {
        match usage {
            ImageUsage::None => {}
            ImageUsage::CopySrc => vk_flags |= vk::FormatFeatureFlags::TRANSFER_SRC,
            ImageUsage::CopyDst => vk_flags |= vk::FormatFeatureFlags::TRANSFER_DST,
            ImageUsage::PipelineAttachment => {
                if format.is_depth() {
                    vk_flags |= vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk_flags |= vk::FormatFeatureFlags::COLOR_ATTACHMENT
                }
            }
            ImageUsage::Present => {}
        }
    }
    vk_flags
}

#[derive(Debug, thiserror::Error)]
pub enum V12Image2dError {
    #[error("Error creating Vulkan Image: {0}")]
    CreateError(vk::Result),
    #[error("Error creating Vulkan Image View: {0}")]
    ViewCreateError(vk::Result),
    #[error("Error with Mutex locks. Addl Context: {0}")]
    LockError(String),
    #[error("Error allocation memory: {0}")]
    AllocationError(#[from] AllocationError),
    #[error("Error binding memory to image: {0}")]
    MemoryBindError(vk::Result),
}

pub struct V12Image2d {
    name: String,
    image: vk::Image,
    res: Resolution2d,
    format: ImageFormat,
    usage: BitFlags<ImageUsage>,
    memory: Option<VkMemAllocation>,
    view: vk::ImageView,
    device: Arc<V12Device>,
}

impl V12Image2d {
    pub fn new(
        device: Arc<V12Device>,
        allocator: &mut VkMemAllocator,
        gpu_local: bool,
        name: &str,
        resolution: Resolution2d,
        format: ImageFormat,
        usage: BitFlags<ImageUsage>,
    ) -> Result<V12Image2d, V12Image2dError> {
        let usage_flags = image_usage_to_vk(usage, format);
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage_flags)
            .format(format_to_vk(format))
            .extent(res_to_extent_3d(resolution))
            .array_layers(1)
            .mip_levels(1)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .tiling(vk::ImageTiling::OPTIMAL);
        let image = unsafe {
            device
                .device
                .create_image(&image_create_info, None)
                .map_err(V12Image2dError::CreateError)?
        };

        let reqs = unsafe { device.device.get_image_memory_requirements(image) };
        let location = if gpu_local {
            MemoryLocation::GpuOnly
        } else {
            MemoryLocation::CpuToGpu
        };
        let mem = allocator
            .allocator
            .lock()
            .map_err(|e| V12Image2dError::LockError(format!("at locking allocator mutex: {e}")))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements: reqs,
                location: location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe {
            device
                .device
                .bind_image_memory(image, mem.memory(), mem.offset())
                .map_err(V12Image2dError::MemoryBindError)?;
        }
        let memory = Some(VkMemAllocation {
            allocation: ManuallyDrop::new(mem),
            allocator: allocator.allocator.clone(),
            is_gpu_local: gpu_local,
        });

        Self::new_wrap(device, name, resolution, format, image, memory, usage)
    }

    pub fn new_wrap(
        device: Arc<V12Device>,
        name: &str,
        resolution: Resolution2d,
        format: ImageFormat,
        image: vk::Image,
        memory: Option<VkMemAllocation>,
        usage: BitFlags<ImageUsage>,
    ) -> Result<Self, V12Image2dError> {
        let view_create_info = vk::ImageViewCreateInfo::default()
            .format(format_to_vk(format))
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(image_2d_subresource_range(format));
        let view = unsafe {
            device
                .device
                .create_image_view(&view_create_info, None)
                .map_err(V12Image2dError::ViewCreateError)?
        };
        Ok(Self {
            name: name.to_string(),
            image,
            res: resolution,
            format,
            usage,
            memory,
            view,
            device,
        })
    }
}

impl Image2d for V12Image2d {
    type AllocatorType = VkMemAllocator;

    type MemType = VkMemAllocation;

    type E = V12Image2dError;

    fn resolution(&self) -> Resolution2d {
        self.res
    }

    fn format(&self) -> ImageFormat {
        self.format
    }

    fn usage(&self) -> BitFlags<ImageUsage> {
        self.usage
    }
}

impl Drop for V12Image2d {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            if self.memory.is_some() {
                self.device.device.destroy_image(self.image, None);
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12SemaphoreError {
    #[error("Error creating vulkan semaphore: {0}")]
    CreateError(vk::Result),
}

pub struct V12Semaphore {
    semaphore: vk::Semaphore,
    device: Arc<V12Device>,
}

impl V12Semaphore {
    pub fn new(device: Arc<V12Device>) -> Result<Self, V12SemaphoreError> {
        let semaphore = unsafe {
            device
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(V12SemaphoreError::CreateError)?
        };
        Ok(Self { semaphore, device })
    }
}

impl GpuFuture for V12Semaphore {}

impl Drop for V12Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12FenceError {
    #[error("Error creating vulkan fence: {0}")]
    CreateError(vk::Result),
    #[error("Error waiting for fence: {0}")]
    WaitError(vk::Result),
    #[error("Error resetting fence: {0}")]
    ResetError(vk::Result),
}

pub struct V12Fence {
    fence: vk::Fence,
    device: Arc<V12Device>,
}

impl V12Fence {
    pub fn new(device: Arc<V12Device>, signaled: bool) -> Result<Self, V12FenceError> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let fence = unsafe {
            device
                .device
                .create_fence(&vk::FenceCreateInfo::default().flags(flags), None)
                .map_err(V12FenceError::CreateError)?
        };
        Ok(Self { fence, device })
    }
}

impl CpuFuture for V12Fence {
    type E = V12FenceError;

    fn wait(&self) -> Result<(), Self::E> {
        unsafe {
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(V12FenceError::WaitError)?;
            self.device
                .device
                .reset_fences(&[self.fence])
                .map_err(V12FenceError::ResetError)?;
        }
        Ok(())
    }
}

impl Drop for V12Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_fence(self.fence, None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12SwapchainError {
    #[error("Error getting surface formats: {0}")]
    GetSurfaceFormatsError(vk::Result),
    #[error("Error getting surface capabilities: {0}")]
    GetSurfaceCapabilitiesError(vk::Result),
    #[error("Error getting present modes: {0}")]
    GetPresentModesError(vk::Result),
    #[error("No supported surface formats found")]
    NoSuitableSurfaceFormat,
    #[error("Error creating Vulkan Swapchain: {0}")]
    CreateError(vk::Result),
    #[error("Error getting swapchain images: {0}")]
    GetImageError(vk::Result),
    #[error("Error wrapping swapchain image into Image2d object: {0}")]
    ImageWrapError(#[from] V12Image2dError),
    #[error("Error with fence operations: {0}")]
    FenceError(#[from] V12FenceError),
    #[error("Error acquiring image from swapchain: {0}")]
    GetNextImageError(vk::Result),
    #[error("Error at presenting image to screen: {0}")]
    PresentError(vk::Result),
}

pub struct V12Swapchain {
    swapchain: vk::SwapchainKHR,
    res: Resolution2d,
    format: vk::SurfaceFormatKHR,
    usages: BitFlags<ImageUsage>,
    images: Vec<V12Image2d>,
    present_mode: vk::PresentModeKHR,
    optimized: bool,
    device: Arc<V12Device>,
}

impl V12Swapchain {
    pub fn new(
        device: Arc<V12Device>,
        usages: BitFlags<ImageUsage>,
    ) -> Result<Self, V12SwapchainError> {
        let surface_instance = &device.loader.surface_instance;
        let surface = device.loader.surface;
        let instance = &device.loader.instance;

        let formats = unsafe {
            surface_instance
                .get_physical_device_surface_formats(device.physical_device, surface)
                .map_err(V12SwapchainError::GetSurfaceFormatsError)?
        };

        let caps = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(device.physical_device, surface)
                .map_err(V12SwapchainError::GetSurfaceCapabilitiesError)?
        };

        let present_modes = unsafe {
            surface_instance
                .get_physical_device_surface_present_modes(device.physical_device, surface)
                .map_err(V12SwapchainError::GetPresentModesError)?
        };

        let format = formats
            .iter()
            .filter(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .filter(|format| {
                // format.format == vk::Format::B8G8R8A8_UNORM
                //     || format.format == vk::Format::R8G8B8A8_UNORM
                //    ||
                format.format == vk::Format::B8G8R8A8_SRGB
                    || format.format == vk::Format::R8G8B8A8_SRGB
            })
            .filter(|format| {
                let supported = unsafe {
                    instance
                        .get_physical_device_format_properties(
                            device.physical_device,
                            format.format,
                        )
                        .optimal_tiling_features
                        .contains(image_usage_to_feature(
                            usages,
                            vk_to_format(format.format).unwrap_or(ImageFormat::Rgba8Srgb),
                        ))
                };
                supported
            })
            .next()
            .cloned()
            .ok_or(V12SwapchainError::NoSuitableSurfaceFormat)?;

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = device.loader.window.inner_size();
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
        let resolution = Resolution2d {
            width: extent.width,
            height: extent.height,
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(swapchain_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(image_usage_to_vk(
                usages,
                vk_to_format(format.format).unwrap_or(ImageFormat::Rgba8Srgb),
            ))
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe {
            device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(V12SwapchainError::CreateError)?
        };

        let images: Vec<_> = unsafe {
            device
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(V12SwapchainError::GetImageError)?
                .into_iter()
                .enumerate()
                .map(|(i, img)| {
                    V12Image2d::new_wrap(
                        device.clone(),
                        &format!("swapchain_image_{i}"),
                        resolution,
                        vk_to_format(format.format).unwrap_or(ImageFormat::Rgba8Srgb),
                        img,
                        None,
                        usages,
                    )
                })
                .collect::<Result<_, _>>()?
        };

        Ok(Self {
            swapchain,
            res: resolution,
            format,
            usages,
            images,
            present_mode,
            optimized: false,
            device,
        })
    }
}

impl Swapchain for V12Swapchain {
    type Image2dType = V12Image2d;

    type GFutType = V12Semaphore;

    type CFutType = V12Fence;

    type E = V12SwapchainError;

    fn is_optimized(&self) -> bool {
        self.optimized
    }

    fn get_next_image(
        &mut self,
        cfut: Option<&Self::CFutType>,
        gfut: Option<&Self::GFutType>,
    ) -> Result<u32, Self::E> {
        loop {
            let aquire_out = unsafe {
                self.device.swapchain_device.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    gfut.map(|x| x.semaphore).unwrap_or(vk::Semaphore::null()),
                    cfut.map(|x| x.fence).unwrap_or(vk::Fence::null()),
                )
            };

            let (idx, is_suboptimal) = match aquire_out {
                Ok((i, s)) => (Some(i), s),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (None, true),
                Err(e) => return Err(V12SwapchainError::GetNextImageError(e)),
            };

            if is_suboptimal {
                self.resize_resolution()?;
                if idx.is_some()
                    && let Some(fence) = cfut
                {
                    fence.wait()?;
                }
                continue;
            }
            if let Some(img_idx) = idx {
                return Ok(img_idx);
            }
        }
    }

    fn resize_resolution(&mut self) -> Result<(), Self::E> {
        let caps = unsafe {
            self.device
                .loader
                .surface_instance
                .get_physical_device_surface_capabilities(
                    self.device.physical_device,
                    self.device.loader.surface,
                )
                .map_err(V12SwapchainError::GetSurfaceCapabilitiesError)?
        };

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = self.device.loader.window.inner_size();
            extent.width = window_res.width;
            extent.height = window_res.height;
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device.loader.surface)
            .min_image_count(self.images.len() as u32)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(image_usage_to_vk(
                self.usages,
                vk_to_format(self.format.format).unwrap_or(ImageFormat::Rgba8Srgb),
            ))
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain);

        let swapchain = unsafe {
            self.device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(V12SwapchainError::CreateError)?
        };

        let resolution = Resolution2d {
            width: extent.width,
            height: extent.height,
        };

        let new_images: Vec<_> = unsafe {
            self.device
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(V12SwapchainError::GetImageError)?
                .into_iter()
                .enumerate()
                .map(|(i, img)| {
                    V12Image2d::new_wrap(
                        self.device.clone(),
                        &format!("swapchain_image_{i}"),
                        resolution,
                        vk_to_format(self.format.format).unwrap_or(ImageFormat::Rgba8Srgb),
                        img,
                        None,
                        self.usages,
                    )
                })
                .collect::<Result<_, _>>()?
        };

        self.images.clear();
        unsafe {
            self.device
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }

        self.swapchain = swapchain;
        self.images = new_images;
        self.res = resolution;
        self.optimized = false;
        Ok(())
    }

    fn images(&self) -> &[Self::Image2dType] {
        &self.images
    }

    fn present(&self, idx: u32, wait_for: &[&Self::GFutType]) -> Result<bool, Self::E> {
        let wait_sems: Vec<_> = wait_for.iter().map(|x| x.semaphore).collect();
        unsafe {
            self.device
                .swapchain_device
                .queue_present(
                    self.device.g_queue.1,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.swapchain])
                        .image_indices(&[idx])
                        .wait_semaphores(&wait_sems),
                )
                .map_err(V12SwapchainError::PresentError)
        }
    }
}

impl Drop for V12Swapchain {
    fn drop(&mut self) {
        self.images.clear();
        unsafe {
            self.device
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

pub enum GpuCommand {
    Image2dOptimize {
        img: vk::Image,
        format: ImageFormat,
        usage: ImageUsage,
    },
    BufferToImage2d {
        buffer: vk::Buffer,
        image: vk::Image,
        image_format: ImageFormat,
        image_size: vk::Extent2D,
    },
    BlitImage2dFull {
        src: vk::Image,
        src_format: ImageFormat,
        src_size: vk::Extent2D,
        dst: vk::Image,
        dst_format: ImageFormat,
        dst_size: vk::Extent2D,
    },
}

impl GpuCommand {
    fn cmd_image_2d_barrier(
        command_buffer: &V12CommandBuffer,
        img: vk::Image,
        format: ImageFormat,
        src_usage: ImageUsage,
        dst_usage: ImageUsage,
    ) {
        unsafe {
            command_buffer.device.device.cmd_pipeline_barrier(
                command_buffer.command_buffer,
                image_usage_to_stage(src_usage),
                image_usage_to_stage(dst_usage),
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(img)
                    .old_layout(image_usage_to_layout(src_usage, format))
                    .new_layout(image_usage_to_layout(dst_usage, format))
                    .src_access_mask(image_usage_to_access(src_usage, format))
                    .dst_access_mask(image_usage_to_access(dst_usage, format))
                    .subresource_range(image_2d_subresource_range(format))],
            );
        }
    }
    pub fn record_commands(commands: &[GpuCommand], command_buffer: &V12CommandBuffer) {
        let mut image_last_states = HashMap::new();
        for command in commands {
            match *command {
                GpuCommand::Image2dOptimize { img, format, usage } => {
                    if let Some(last_usage) = image_last_states.get_mut(&img) {
                        if *last_usage == usage {
                            continue;
                        }
                        Self::cmd_image_2d_barrier(command_buffer, img, format, *last_usage, usage);
                        *last_usage = usage;
                    } else {
                        image_last_states.insert(img, usage);
                    }
                }
                GpuCommand::BufferToImage2d {
                    buffer,
                    image,
                    image_format,
                    image_size,
                } => {
                    if let Some(last_usage) = image_last_states.get_mut(&image) {
                        if *last_usage == ImageUsage::CopyDst {
                            continue;
                        }
                        Self::cmd_image_2d_barrier(
                            command_buffer,
                            image,
                            image_format,
                            *last_usage,
                            ImageUsage::CopyDst,
                        );
                        *last_usage = ImageUsage::CopyDst;
                    } else {
                        image_last_states.insert(image, ImageUsage::CopyDst);
                    }
                    unsafe {
                        command_buffer.device.device.cmd_copy_buffer_to_image(
                            command_buffer.command_buffer,
                            buffer,
                            image,
                            image_usage_to_layout(ImageUsage::CopyDst, image_format),
                            &[vk::BufferImageCopy::default()
                                .image_offset(vk::Offset3D::default())
                                .image_extent(vk::Extent3D::from(image_size).depth(1))
                                .image_subresource(image_2d_subresource_layers(image_format))],
                        );
                    }
                }
                GpuCommand::BlitImage2dFull {
                    src,
                    src_format,
                    src_size,
                    dst,
                    dst_format,
                    dst_size,
                } => {
                    if let Some(last_usage) = image_last_states.get_mut(&src) {
                        if *last_usage == ImageUsage::CopySrc {
                            continue;
                        }
                        Self::cmd_image_2d_barrier(
                            command_buffer,
                            src,
                            src_format,
                            *last_usage,
                            ImageUsage::CopySrc,
                        );
                        *last_usage = ImageUsage::CopySrc;
                    } else {
                        image_last_states.insert(src, ImageUsage::CopySrc);
                    }

                    if let Some(last_usage) = image_last_states.get_mut(&dst) {
                        if *last_usage == ImageUsage::CopyDst {
                            continue;
                        }
                        Self::cmd_image_2d_barrier(
                            command_buffer,
                            dst,
                            dst_format,
                            *last_usage,
                            ImageUsage::CopyDst,
                        );
                        *last_usage = ImageUsage::CopyDst;
                    } else {
                        image_last_states.insert(dst, ImageUsage::CopyDst);
                    }

                    unsafe {
                        command_buffer.device.device.cmd_blit_image(
                            command_buffer.command_buffer,
                            src,
                            image_usage_to_layout(ImageUsage::CopySrc, src_format),
                            dst,
                            image_usage_to_layout(ImageUsage::CopyDst, dst_format),
                            &[vk::ImageBlit::default()
                                .src_offsets([
                                    vk::Offset3D::default(),
                                    vk::Offset3D {
                                        x: src_size.width as _,
                                        y: src_size.height as _,
                                        z: 1,
                                    },
                                ])
                                .dst_offsets([
                                    vk::Offset3D::default(),
                                    vk::Offset3D {
                                        x: dst_size.width as _,
                                        y: dst_size.height as _,
                                        z: 1,
                                    },
                                ])
                                .src_subresource(image_2d_subresource_layers(src_format))
                                .dst_subresource(image_2d_subresource_layers(dst_format))],
                            vk::Filter::NEAREST,
                        );
                    }
                }
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12CommandBufferError {
    #[error("Error creating Vulkan Command Buffer: {0}")]
    CreateError(vk::Result),
    #[error("Error resetting Command Buffer: {0}")]
    ResetError(vk::Result),
    #[error("Error beginning Command Buffer: {0}")]
    BeginError(vk::Result),
    #[error("Error ending Command Buffer: {0}")]
    EndError(vk::Result),
    #[error("Error submitting Command Buffer: {0}")]
    SubmitError(vk::Result),
}

pub struct V12CommandBuffer {
    wait_sems: Vec<vk::Semaphore>,
    emit_sems: Vec<vk::Semaphore>,
    emit_fence: Option<vk::Fence>,
    command_to_record: Vec<GpuCommand>,
    command_buffer: vk::CommandBuffer,
    queue_type: QueueType,
    device: Arc<V12Device>,
}

impl V12CommandBuffer {
    pub fn new(
        device: Arc<V12Device>,
        command_pool: vk::CommandPool,
        queue_type: QueueType,
    ) -> Result<Self, V12CommandBufferError> {
        let command_buffer = unsafe {
            device
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .map_err(V12CommandBufferError::CreateError)?
                .remove(0)
        };
        Ok(Self {
            wait_sems: vec![],
            emit_sems: vec![],
            emit_fence: None,
            command_to_record: vec![],
            command_buffer,
            queue_type,
            device,
        })
    }
}

impl CommandBuffer for V12CommandBuffer {
    type BufferType = V12Buffer;

    type Image2dType = V12Image2d;

    type GFutType = V12Semaphore;

    type CFutType = V12Fence;

    type E = V12CommandBufferError;

    fn queue_type(&self) -> QueueType {
        self.queue_type
    }

    fn add_image_2d_optimize_cmd(&mut self, image: &Self::Image2dType, usage: ImageUsage) {
        self.command_to_record.push(GpuCommand::Image2dOptimize {
            img: image.image,
            format: image.format,
            usage,
        });
    }

    fn copy_buffer_to_image_2d_cmd(
        &mut self,
        buffer: &Self::BufferType,
        image: &Self::Image2dType,
    ) {
        self.command_to_record.push(GpuCommand::BufferToImage2d {
            buffer: buffer.buffer,
            image: image.image,
            image_format: image.format,
            image_size: res_to_extent_2d(image.res),
        });
    }

    fn add_blit_image_2d_cmd(&mut self, src: &Self::Image2dType, dst: &Self::Image2dType) {
        self.command_to_record.push(GpuCommand::BlitImage2dFull {
            src: src.image,
            src_format: src.format,
            src_size: res_to_extent_2d(src.res),
            dst: dst.image,
            dst_format: dst.format,
            dst_size: res_to_extent_2d(dst.res),
        });
    }

    fn build(&mut self) -> Result<(), Self::E> {
        unsafe {
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(V12CommandBufferError::BeginError)?;
        }
        GpuCommand::record_commands(&self.command_to_record, &self);
        unsafe {
            self.device
                .device
                .end_command_buffer(self.command_buffer)
                .map_err(V12CommandBufferError::EndError)?;
        }
        self.command_to_record.clear();
        Ok(())
    }

    fn reset(&mut self) -> Result<(), Self::E> {
        self.command_to_record.clear();
        self.emit_sems.clear();
        self.wait_sems.clear();
        self.emit_fence.take();
        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(V12CommandBufferError::ResetError)
        }
    }

    fn add_wait_for_gpu_future(&mut self, fut: &Self::GFutType) {
        self.wait_sems.push(fut.semaphore);
    }

    fn emit_gpu_future_on_finish(&mut self, fut: &Self::GFutType) {
        self.emit_sems.push(fut.semaphore);
    }

    fn emit_cpu_future_on_finish(&mut self, fut: &Self::CFutType) {
        self.emit_fence.replace(fut.fence);
    }

    fn submit(&self) -> Result<(), Self::E> {
        let wait_stages: Vec<_> = (0..self.emit_sems.len())
            .map(|_| vk::PipelineStageFlags::ALL_COMMANDS)
            .collect();
        let cmd_buffers = [self.command_buffer];
        let mut submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        if self.emit_sems.len() > 0 {
            submit_info = submit_info.signal_semaphores(&self.emit_sems);
        };
        if self.wait_sems.len() > 0 {
            submit_info = submit_info
                .wait_semaphores(&self.wait_sems)
                .wait_dst_stage_mask(&wait_stages);
        };
        let queue = self.device.g_queue.1;
        unsafe {
            self.device
                .device
                .queue_submit(
                    queue,
                    &[submit_info],
                    self.emit_fence.unwrap_or(vk::Fence::null()),
                )
                .map_err(V12CommandBufferError::SubmitError)?;
        }
        Ok(())
    }
}

pub struct V12Device {
    physical_device: vk::PhysicalDevice,
    g_queue: (u32, vk::Queue),
    swapchain_device: khr::swapchain::Device,
    device: ash::Device,
    loader: V12ApiLoader,
}

impl Drop for V12Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12ContextError {
    #[error("Buffer related error: {0}")]
    BufferError(#[from] V12BufferError),
    #[error("2D Image related error: {0}")]
    Image2dError(#[from] V12Image2dError),
    #[error("Allocator related error: {0}")]
    AllocatorError(AllocationError),
    #[error("Swapchain related error: {0}")]
    SwapchainError(#[from] V12SwapchainError),
    #[error("Semaphore related error: {0}")]
    SemaphoreError(#[from] V12SemaphoreError),
    #[error("Fence related error: {0}")]
    FenceError(#[from] V12FenceError),
    #[error("Command buffer related error: {0}")]
    CommandBufferError(#[from] V12CommandBufferError),
    #[error("'image' library related error: {0}")]
    ImageLibError(#[from] image::ImageError),
}

pub struct V12Context {
    command_pool: vk::CommandPool,
    device: Arc<V12Device>,
}

impl GpuContext for V12Context {
    type AllocatorType = VkMemAllocator;

    type AllocationType = VkMemAllocation;

    type BufferType = V12Buffer;

    type Image2dType = V12Image2d;

    type SwapchainType = V12Swapchain;

    type CommandBufferType = V12CommandBuffer;

    type GFutType = V12Semaphore;

    type CFutType = V12Fence;

    type E = V12ContextError;

    fn new_buffer(
        &self,
        allocator: &mut Self::AllocatorType,
        gpu_local: bool,
        size: u64,
        name: &str,
        usage: BitFlags<BufferUsage>,
    ) -> Result<Self::BufferType, Self::E> {
        let buffer = V12Buffer::new(self.device.clone(), allocator, gpu_local, name, size, usage)?;
        Ok(buffer)
    }

    fn new_image_2d(
        &self,
        allocator: &mut Self::AllocatorType,
        gpu_local: bool,
        name: &str,
        resolution: Resolution2d,
        format: ImageFormat,
        usage: BitFlags<ImageUsage>,
    ) -> Result<Self::Image2dType, Self::E> {
        let image = V12Image2d::new(
            self.device.clone(),
            allocator,
            gpu_local,
            name,
            resolution,
            format,
            usage,
        )?;
        Ok(image)
    }

    fn new_allocator(&self) -> Result<Self::AllocatorType, Self::E> {
        VkMemAllocator::new(
            &self.device.device,
            &self.device.loader.instance,
            self.device.physical_device,
        )
        .map_err(V12ContextError::AllocatorError)
    }

    fn new_swapchain(&self, usages: BitFlags<ImageUsage>) -> Result<Self::SwapchainType, Self::E> {
        let swapchain = V12Swapchain::new(self.device.clone(), usages)?;
        Ok(swapchain)
    }

    fn new_gpu_future(&self) -> Result<Self::GFutType, Self::E> {
        let sem = V12Semaphore::new(self.device.clone())?;
        Ok(sem)
    }

    fn new_cpu_future(&self, signalled: bool) -> Result<Self::CFutType, Self::E> {
        let fence = V12Fence::new(self.device.clone(), signalled)?;
        Ok(fence)
    }

    fn new_command_buffer(
        &self,
        queue_type: QueueType,
    ) -> Result<Self::CommandBufferType, Self::E> {
        let command_buffer =
            V12CommandBuffer::new(self.device.clone(), self.command_pool, queue_type)?;
        Ok(command_buffer)
    }
}

impl Drop for V12Context {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct V12GpuInfo {
    physical_device: vk::PhysicalDevice,
    props: vk::PhysicalDeviceProperties,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    g_queue_family: (usize, vk::QueueFamilyProperties),
}

impl GpuInfo for V12GpuInfo {
    fn name(&self) -> String {
        self.props
            .device_name_as_c_str()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or("Unknown Device Name".to_string())
    }

    fn vram(&self) -> u64 {
        self.mem_props
            .memory_heaps_as_slice()
            .iter()
            .filter(|x| x.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|x| x.size)
            .sum()
    }

    fn is_dedicated(&self) -> bool {
        self.props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12ApiLoaderError {
    #[error("Error loading Vulkan: {0}")]
    VkLoadError(#[from] ash::LoadingError),
    #[error("Error creating Vulkan Instance: {0}")]
    InstanceCreateError(vk::Result),
    #[error("Error getting window's handle: {0}")]
    WindowHandleError(#[from] HandleError),
    #[error("Error creating Vulkan Surface: {0}")]
    SurfaceCreationError(vk::Result),
    #[error("Error creating Vulkan Device: {0}")]
    DeviceCreateError(vk::Result),
    #[error("Error creating Vulkan Command Pool: {0}")]
    CommandPoolCreateError(vk::Result),
}

pub struct V12ApiLoader {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_instance: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    window: winit::window::Window,
}

impl V12ApiLoader {
    pub fn new(window: winit::window::Window) -> Result<Self, V12ApiLoaderError> {
        let entry = unsafe { ash::Entry::load()? };
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2)
            .application_name(c"Caugher App")
            .application_version(1)
            .engine_name(c"Caugher Vulkan 1.2")
            .engine_version(1);
        let layers = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation".as_ptr(),
        ];
        let extensions = [
            #[cfg(debug_assertions)]
            ext::debug_utils::NAME.as_ptr(),
            khr::surface::NAME.as_ptr(),
            #[cfg(target_os = "windows")]
            khr::win32_surface::NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            khr::xlib_surface::NAME.as_ptr(),
            // #[cfg(target_os = "linux")]
            // khr::wayland_surface::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            khr::portability_enumeration::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            ext::metal_surface::NAME.as_ptr(),
            #[cfg(target_os = "android")]
            khr::android_surface::NAME.as_ptr(),
        ];

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
                .map_err(V12ApiLoaderError::InstanceCreateError)?
        };

        let surface_instance = khr::surface::Instance::new(&entry, &instance);

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .map_err(V12ApiLoaderError::SurfaceCreationError)?
        };

        Ok(Self {
            _entry: entry,
            instance,
            surface_instance,
            surface,
            window,
        })
    }
}

impl ApiLoader for V12ApiLoader {
    type GpuInfoType = V12GpuInfo;

    type ContextType = V12Context;

    type E = V12ApiLoaderError;

    fn list_supported_gpus(&self) -> Vec<Self::GpuInfoType> {
        let gpus = unsafe { self.instance.enumerate_physical_devices().unwrap_or(vec![]) };
        gpus.into_iter()
            .filter_map(|g| unsafe {
                let props = self.instance.get_physical_device_properties(g);
                let mem_props = self.instance.get_physical_device_memory_properties(g);
                let g_queue_idx = self
                    .instance
                    .get_physical_device_queue_family_properties(g)
                    .into_iter()
                    .enumerate()
                    .filter(|(_, qfp)| qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .filter(|(qid, _)| {
                        self.surface_instance
                            .get_physical_device_surface_support(g, *qid as _, self.surface)
                            .unwrap_or(false)
                    })
                    .min_by_key(|x| x.1.queue_count)?;
                Some(V12GpuInfo {
                    physical_device: g,
                    props,
                    mem_props,
                    g_queue_family: g_queue_idx,
                })
            })
            .collect()
    }

    fn new_gpu_context(self, gpu: Self::GpuInfoType) -> Result<Self::ContextType, Self::E> {
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
            self.instance
                .create_device(gpu.physical_device, &device_create_info, None)
                .map_err(V12ApiLoaderError::DeviceCreateError)?
        };
        let g_queue = unsafe { device.get_device_queue(gpu.g_queue_family.0 as _, 0) };

        let swapchain_device = khr::swapchain::Device::new(&self.instance, &device);

        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(gpu.g_queue_family.0 as u32)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(V12ApiLoaderError::CommandPoolCreateError)?
        };

        Ok(V12Context {
            command_pool,
            device: Arc::new(V12Device {
                physical_device: gpu.physical_device,
                g_queue: (gpu.g_queue_family.0 as _, g_queue),
                swapchain_device,
                device,
                loader: self,
            }),
        })
    }
}

impl Drop for V12ApiLoader {
    fn drop(&mut self) {
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
