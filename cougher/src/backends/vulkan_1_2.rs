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
        VkMemAllocation, VkMemAllocator, binding_type_to_vk, buffer_usage_to_vk,
        format_has_stencil, format_to_aspect_mask, format_to_vk, image_2d_subresource_range,
        is_format_depth, res_to_extent_3d, shader_type_to_vk,
    },
    traits::{
        ApiLoader, Buffer, BufferUsage, CpuFuture, GpuCommand, GpuContext, GpuExecutor, GpuFuture,
        GpuInfo, GraphicsPass, GraphicsPassAttachments, GraphicsPassCommand, Image2d, ImageFormat,
        ImageUsage, PipelineSet, PipelineSetBindingInfo, PipelineSetBindingType,
        PipelineSetBindingWritable, QueueType, Resolution2d, ShaderType, SubpassInfo, Swapchain,
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
    pub(crate) name: String,
    pub(crate) buffer: vk::Buffer,
    pub(crate) size: u64,
    pub(crate) usage: BitFlags<BufferUsage>,
    pub(crate) memory: Option<VkMemAllocation>,
    pub(crate) device: Arc<V12Device>,
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

pub fn image_usage_to_layout(usage: ImageUsage, format: vk::Format) -> vk::ImageLayout {
    match usage {
        ImageUsage::None => vk::ImageLayout::UNDEFINED,
        ImageUsage::CopySrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageUsage::CopyDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageUsage::PipelineAttachment => {
            if is_format_depth(format) {
                if format_has_stencil(format) {
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

pub fn image_usage_to_access(usage: ImageUsage, format: vk::Format) -> vk::AccessFlags {
    match usage {
        ImageUsage::None => vk::AccessFlags::empty(),
        ImageUsage::CopySrc => vk::AccessFlags::TRANSFER_READ,
        ImageUsage::CopyDst => vk::AccessFlags::TRANSFER_WRITE,
        ImageUsage::PipelineAttachment => {
            if is_format_depth(format) {
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

pub fn image_usage_to_vk(usages: BitFlags<ImageUsage>, format: vk::Format) -> vk::ImageUsageFlags {
    let mut vk_flags = vk::ImageUsageFlags::default();
    for usage in usages {
        match usage {
            ImageUsage::None => {}
            ImageUsage::CopySrc => vk_flags |= vk::ImageUsageFlags::TRANSFER_SRC,
            ImageUsage::CopyDst => vk_flags |= vk::ImageUsageFlags::TRANSFER_DST,
            ImageUsage::PipelineAttachment => {
                if is_format_depth(format) {
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

pub fn usage_needs_image_view(usages: BitFlags<ImageUsage>) -> bool {
    for usage in usages {
        match usage {
            ImageUsage::PipelineAttachment => return true,
            _ => {}
        }
    }
    false
}

pub fn image_usage_to_feature(
    usages: BitFlags<ImageUsage>,
    format: vk::Format,
) -> vk::FormatFeatureFlags {
    let mut vk_flags = vk::FormatFeatureFlags::default();
    for usage in usages {
        match usage {
            ImageUsage::None => {}
            ImageUsage::CopySrc => vk_flags |= vk::FormatFeatureFlags::TRANSFER_SRC,
            ImageUsage::CopyDst => vk_flags |= vk::FormatFeatureFlags::TRANSFER_DST,
            ImageUsage::PipelineAttachment => {
                if is_format_depth(format) {
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
    res: vk::Extent2D,
    format: vk::Format,
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
        let vk_format = format_to_vk(format);
        let usage_flags = image_usage_to_vk(usage, vk_format);
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage_flags)
            .format(vk_format)
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

        let extent = vk::Extent2D {
            width: resolution.width,
            height: resolution.height,
        };

        Self::new_wrap(device, name, extent, vk_format, image, memory, usage)
    }

    pub fn new_wrap(
        device: Arc<V12Device>,
        name: &str,
        resolution: vk::Extent2D,
        format: vk::Format,
        image: vk::Image,
        memory: Option<VkMemAllocation>,
        usage: BitFlags<ImageUsage>,
    ) -> Result<Self, V12Image2dError> {
        let view = if usage_needs_image_view(usage) {
            let view_create_info = vk::ImageViewCreateInfo::default()
                .format(format)
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(image_2d_subresource_range(format));
            unsafe {
                device
                    .device
                    .create_image_view(&view_create_info, None)
                    .map_err(V12Image2dError::ViewCreateError)?
            }
        } else {
            vk::ImageView::null()
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

    pub fn extent_3d(&self) -> vk::Extent3D {
        vk::Extent3D::default()
            .width(self.res.width)
            .height(self.res.height)
            .depth(1)
    }

    pub fn full_size_offset(&self) -> [vk::Offset3D; 2] {
        [
            vk::Offset3D::default(),
            vk::Offset3D::default()
                .x(self.res.width as _)
                .y(self.res.height as _)
                .z(1),
        ]
    }

    pub fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(format_to_aspect_mask(self.format))
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0)
    }
}

impl Image2d for V12Image2d {
    type AllocatorType = VkMemAllocator;

    type MemType = VkMemAllocation;

    type E = V12Image2dError;

    fn resolution(&self) -> Resolution2d {
        Resolution2d {
            width: self.res.width,
            height: self.res.height,
        }
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
                format.format == vk::Format::B8G8R8A8_UNORM
                    || format.format == vk::Format::R8G8B8A8_UNORM
                    || format.format == vk::Format::B8G8R8A8_SRGB
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
                        .contains(image_usage_to_feature(usages, format.format))
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
            .image_usage(image_usage_to_vk(usages, format.format))
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
                        extent,
                        format.format,
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
            .image_usage(image_usage_to_vk(self.usages, self.format.format))
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
                        extent,
                        self.format.format,
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
                    self.device.queues[&QueueType::Graphics].1,
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

pub struct V12GPassAttachments {
    res: Resolution2d,
    framebuffer: vk::Framebuffer,
    attachments: Vec<V12Image2d>,
    device: Arc<V12Device>,
}

impl GraphicsPassAttachments for V12GPassAttachments {
    type I2dType = V12Image2d;

    fn get_attachments(&self) -> Vec<&Self::I2dType> {
        self.attachments.iter().collect()
    }

    fn resolution(&self) -> Resolution2d {
        self.res
    }
}

impl Drop for V12GPassAttachments {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_framebuffer(self.framebuffer, None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12PipelineSetError {
    #[error("Error writing set bindings: {0}")]
    SetWriteError(vk::Result),
}

pub struct V12PipelineSet {
    set: vk::DescriptorSet,
    bindings: Vec<PipelineSetBindingInfo>,
    device: Arc<V12Device>,
}

impl PipelineSet for V12PipelineSet {
    type BType = V12Buffer;

    type I2dType = V12Image2d;

    type E = V12PipelineSetError;

    fn update_bindings(
        &mut self,
        binding_writables: Vec<PipelineSetBindingWritable<Self::BType, Self::I2dType>>,
    ) -> Result<(), Self::E> {
        let write_infos: Vec<_> = binding_writables
            .iter()
            .map(|b_writes| match b_writes {
                PipelineSetBindingWritable::Buffer(items) => (
                    items
                        .iter()
                        .map(|x| {
                            vk::DescriptorBufferInfo::default()
                                .buffer(x.buffer)
                                .offset(0)
                                .range(vk::WHOLE_SIZE)
                        })
                        .collect(),
                    vec![],
                ),
                PipelineSetBindingWritable::Image2d(items) => (
                    vec![],
                    items
                        .iter()
                        .map(|x| {
                            vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(x.view)
                        })
                        .collect(),
                ),
            })
            .collect();
        let writes: Vec<_> = write_infos
            .iter()
            .zip(self.bindings.iter())
            .enumerate()
            .map(|(b_id, (write_info, binding_info))| {
                let mut write_vk = vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(b_id as _)
                    .descriptor_type(binding_type_to_vk(binding_info._type));

                if write_info.0.len() > 0 {
                    write_vk = write_vk.buffer_info(&write_info.0);
                }
                if write_info.1.len() > 0 {
                    write_vk = write_vk.image_info(&write_info.1);
                }
                write_vk
            })
            .collect();
        unsafe {
            self.device.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12GraphicsPassError {
    #[error("Error creating Vulkan Render Pass: {0}")]
    RenderPassCreateError(vk::Result),
    #[error("Error creating Vulkan Descriptor Set Layout: {0}")]
    SetLayoutCreateError(vk::Result),
    #[error("Error creating Vulkan Descriptor Pool: {0}")]
    SetPoolCreateError(vk::Result),
    #[error("Error creating Vulkan Pipeline Layout: {0}")]
    PipelineLayoutCreateError(vk::Result),
    #[error("Error creating Vulkan Shader Module: {0}")]
    ShaderLoadError(vk::Result),
    #[error("Error creating Vulkan Pipeline: {0}")]
    PipelineCreateError(vk::Result),
    #[error("Error creating Graphics Pass attachment: {0}")]
    AttachmentCreateError(#[from] V12Image2dError),
    #[error("Error creating Vulkan Framebuffer: {0}")]
    FramebufferCreateError(vk::Result),
    #[error("Error allocating Vulkan Descriptor Sets: {0}")]
    SetAllocateError(vk::Result),
}

pub struct V12GraphicsPass {
    render_pass: vk::RenderPass,
    pipelines: Vec<vk::Pipeline>,
    pipeline_layouts: Vec<vk::PipelineLayout>,
    dsls: Vec<Vec<vk::DescriptorSetLayout>>,
    desc_pool: vk::DescriptorPool,
    subpass_infos: Vec<SubpassInfo>,
    attachment_formats: Vec<ImageFormat>,
    device: Arc<V12Device>,
}

impl V12GraphicsPass {
    fn make_render_pass(
        device: &V12Device,
        attachments: &[ImageFormat],
        subpass_infos: &[SubpassInfo],
    ) -> Result<vk::RenderPass, V12GraphicsPassError> {
        let attach_descs: Vec<_> = attachments
            .iter()
            .map(|&x| {
                let vk_fmt = format_to_vk(x);
                let vk_layout = image_usage_to_layout(ImageUsage::PipelineAttachment, vk_fmt);
                let store_op = if x.is_depth() {
                    vk::AttachmentStoreOp::DONT_CARE
                } else {
                    vk::AttachmentStoreOp::STORE
                };
                vk::AttachmentDescription::default()
                    .initial_layout(vk_layout)
                    .final_layout(vk_layout)
                    .format(vk_fmt)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(store_op)
            })
            .collect();
        let color_refs: Vec<Vec<_>> = subpass_infos
            .iter()
            .enumerate()
            .map(|(s_i, s)| {
                s.color_attachments
                    .iter()
                    .map(|&i| {
                        let vk_fmt = format_to_vk(attachments[i]);
                        let vk_layout =
                            image_usage_to_layout(ImageUsage::PipelineAttachment, vk_fmt);
                        vk::AttachmentReference::default()
                            .layout(vk_layout)
                            .attachment(i as _)
                    })
                    .collect()
            })
            .collect();
        let depth_refs: Vec<Option<_>> = subpass_infos
            .iter()
            .map(|s| {
                s.depth_attachment.map(|i| {
                    let vk_fmt = format_to_vk(attachments[i]);
                    let vk_layout = image_usage_to_layout(ImageUsage::PipelineAttachment, vk_fmt);
                    vk::AttachmentReference::default()
                        .layout(vk_layout)
                        .attachment(i as _)
                })
            })
            .collect();
        let subpass_descs: Vec<_> = subpass_infos
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut desc = vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
                if color_refs[i].len() > 0 {
                    desc = desc.color_attachments(&color_refs[i]);
                }
                if let Some(d_img) = depth_refs[i].as_ref() {
                    desc = desc.depth_stencil_attachment(d_img);
                }
                desc
            })
            .collect();
        let rp_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attach_descs)
            .subpasses(&subpass_descs);
        unsafe {
            device
                .device
                .create_render_pass(&rp_create_info, None)
                .map_err(V12GraphicsPassError::RenderPassCreateError)
        }
    }

    fn make_set_layout(
        device: &V12Device,
        bindings: &[PipelineSetBindingInfo],
    ) -> Result<vk::DescriptorSetLayout, V12GraphicsPassError> {
        let bindings: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .descriptor_type(binding_type_to_vk(b._type))
                    .descriptor_count(b.count as _)
            })
            .collect();
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            device
                .device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(V12GraphicsPassError::SetLayoutCreateError)
        }
    }

    fn make_pipeline_layout(
        device: &V12Device,
        sets: &[vk::DescriptorSetLayout],
    ) -> Result<vk::PipelineLayout, V12GraphicsPassError> {
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
        unsafe {
            device
                .device
                .create_pipeline_layout(&create_info, None)
                .map_err(V12GraphicsPassError::PipelineLayoutCreateError)
        }
    }

    fn load_shader(
        device: &V12Device,
        code: &[u32],
    ) -> Result<vk::ShaderModule, V12GraphicsPassError> {
        unsafe {
            device
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
                .map_err(V12GraphicsPassError::ShaderLoadError)
        }
    }

    fn make_pipeline(
        device: &V12Device,
        render_pass: vk::RenderPass,
        subpass_id: usize,
        subpass_info: &SubpassInfo,
        layout: vk::PipelineLayout,
        shaders: &HashMap<ShaderType, vk::ShaderModule>,
    ) -> Result<vk::Pipeline, V12GraphicsPassError> {
        let vertex_state = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
        let vp_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK);
        let stages: Vec<_> = shaders
            .iter()
            .map(|(stg, shader)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(shader_type_to_vk(*stg))
                    .module(*shader)
                    .name(c"main")
            })
            .collect();
        let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);
        let mut create_info = vk::GraphicsPipelineCreateInfo::default()
            .render_pass(render_pass)
            .subpass(subpass_id as _)
            .layout(layout)
            .vertex_input_state(&vertex_state)
            .input_assembly_state(&input_assembly)
            .dynamic_state(&dyn_state)
            .viewport_state(&vp_state)
            .rasterization_state(&raster_state)
            .stages(&stages);
        if subpass_info.depth_attachment.is_some() {
            create_info = create_info.depth_stencil_state(&depth_state);
        }
        let pipeline = unsafe {
            device
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .map_err(|(_, e)| V12GraphicsPassError::PipelineCreateError(e))?
                .remove(0)
        };
        Ok(pipeline)
    }

    pub fn new(
        device: Arc<V12Device>,
        attachments: Vec<ImageFormat>,
        subpass_infos: Vec<SubpassInfo>,
        max_sets: usize,
    ) -> Result<Self, V12GraphicsPassError> {
        let render_pass = Self::make_render_pass(&device, &attachments, &subpass_infos)?;
        let set_layouts: Vec<Vec<_>> = subpass_infos
            .iter()
            .map(|s| {
                s.set_infos
                    .iter()
                    .map(|sb| Self::make_set_layout(&device, sb))
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<_, _>>()?;
        let pipeline_layouts: Vec<_> = set_layouts
            .iter()
            .map(|sl| Self::make_pipeline_layout(&device, sl))
            .collect::<Result<_, _>>()?;
        let shaders: Vec<HashMap<_, _>> = subpass_infos
            .iter()
            .map(|s| {
                s.shaders
                    .iter()
                    .map(|(st, sc)| {
                        let sm = Self::load_shader(&device, sc)?;
                        Ok::<_, V12GraphicsPassError>((*st, sm))
                    })
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<_, _>>()?;
        let pipelines: Vec<_> = (0..subpass_infos.len())
            .map(|i| {
                Self::make_pipeline(
                    &device,
                    render_pass,
                    i,
                    &subpass_infos[i],
                    pipeline_layouts[i],
                    &shaders[i],
                )
            })
            .collect::<Result<_, _>>()?;

        let mut uniform_buffer_count = 0;
        let mut storage_buffer_count = 0;
        let mut sampler_2d_count = 0;
        for sd in &subpass_infos {
            for psd in &sd.set_infos {
                for bd in psd {
                    match bd._type {
                        PipelineSetBindingType::UniformBuffer => uniform_buffer_count += bd.count,
                        PipelineSetBindingType::StorageBuffer => storage_buffer_count += bd.count,
                        PipelineSetBindingType::Sampler2d => sampler_2d_count += bd.count,
                    }
                }
            }
        }
        uniform_buffer_count *= max_sets;
        storage_buffer_count *= max_sets;

        let desc_pool = unsafe {
            device
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets as _)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(binding_type_to_vk(PipelineSetBindingType::UniformBuffer))
                                .descriptor_count(uniform_buffer_count as _),
                            vk::DescriptorPoolSize::default()
                                .ty(binding_type_to_vk(PipelineSetBindingType::StorageBuffer))
                                .descriptor_count(storage_buffer_count as _),
                            vk::DescriptorPoolSize::default()
                                .ty(binding_type_to_vk(PipelineSetBindingType::StorageBuffer))
                                .descriptor_count(sampler_2d_count as _),
                        ]),
                    None,
                )
                .map_err(V12GraphicsPassError::SetPoolCreateError)?
        };

        Ok(Self {
            render_pass,
            pipelines,
            pipeline_layouts,
            dsls: set_layouts,
            desc_pool,
            subpass_infos,
            attachment_formats: attachments,
            device,
        })
    }
}

impl GraphicsPass for V12GraphicsPass {
    type AllocatorType = VkMemAllocator;

    type MemType = VkMemAllocation;

    type BType = V12Buffer;

    type I2dType = V12Image2d;

    type PSetType = V12PipelineSet;

    type PAttachType = V12GPassAttachments;

    type E = V12GraphicsPassError;

    fn create_sets(&self, subpass_id: usize) -> Result<Vec<Self::PSetType>, Self::E> {
        let sets = unsafe {
            self.device
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.desc_pool)
                        .set_layouts(&self.dsls[subpass_id]),
                )
                .map_err(V12GraphicsPassError::SetAllocateError)?
        };

        Ok(sets
            .into_iter()
            .enumerate()
            .map(|(i, set)| V12PipelineSet {
                set,
                bindings: self.subpass_infos[subpass_id].set_infos[i].clone(),
                device: self.device.clone(),
            })
            .collect())
    }

    fn create_attachments(
        &self,
        name: &str,
        allocator: &mut Self::AllocatorType,
        res: Resolution2d,
    ) -> Result<Self::PAttachType, Self::E> {
        let attachments: Vec<_> = self
            .attachment_formats
            .iter()
            .map(|&fmt| {
                V12Image2d::new(
                    self.device.clone(),
                    allocator,
                    true,
                    name,
                    res,
                    fmt,
                    ImageUsage::CopySrc | ImageUsage::PipelineAttachment,
                )
            })
            .collect::<Result<_, _>>()?;

        let framebuffer = unsafe {
            let attachment_views: Vec<_> = attachments.iter().map(|img| img.view).collect();
            self.device
                .device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .width(res.width)
                        .height(res.width)
                        .layers(1)
                        .render_pass(self.render_pass)
                        .attachments(&attachment_views),
                    None,
                )
                .map_err(V12GraphicsPassError::FramebufferCreateError)?
        };
        Ok(V12GPassAttachments {
            framebuffer,
            res,
            attachments,
            device: self.device.clone(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V12ExecutorError {
    #[error("Queue Type Unsupported")]
    UnsupportedQueue,
    #[error("Error creating Vulkan Command Pool: {0}")]
    CommandPoolCreateError(vk::Result),
    #[error("Error creating Vulkan Command Buffer: {0}")]
    CommandBufferCreateError(vk::Result),
    #[error("Unknown Command Buffer Name: {0}")]
    UnknownCommandBuffer(String),
    #[error("Error submitting work to Vulkan Queue: {0}")]
    QueueSubmissionError(vk::Result),
    #[error("Error starting Vulkan Command Buffer recording: {0}")]
    CommandBufferBeginError(vk::Result),
    #[error("Error ending Vulkan Command Buffer recording: {0}")]
    CommandBufferEndError(vk::Result),
}

pub struct V12Executor {
    pub(crate) type_: QueueType,
    pub(crate) queue: vk::Queue,
    pub(crate) qf_id: u32,
    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) cmd_buffers: HashMap<String, vk::CommandBuffer>,
    pub(crate) device: Arc<V12Device>,
}

impl V12Executor {
    pub fn new(device: Arc<V12Device>, type_: QueueType) -> Result<Self, V12ExecutorError> {
        let (qf_id, queue) = device
            .queues
            .get(&type_)
            .cloned()
            .ok_or(V12ExecutorError::UnsupportedQueue)?;
        let cmd_pool = unsafe {
            device
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qf_id)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(V12ExecutorError::CommandPoolCreateError)?
        };
        Ok(Self {
            type_,
            queue,
            qf_id,
            cmd_pool,
            cmd_buffers: HashMap::default(),
            device,
        })
    }

    fn cmd_image_2d_barrier(
        device: &V12Device,
        cmd_buffer: vk::CommandBuffer,
        img: vk::Image,
        format: vk::Format,
        src_usage: ImageUsage,
        dst_usage: ImageUsage,
    ) {
        unsafe {
            device.device.cmd_pipeline_barrier(
                cmd_buffer,
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

    fn update_image_usage(
        state: &mut HashMap<vk::Image, ImageUsage>,
        device: &V12Device,
        cmd_buffer: vk::CommandBuffer,
        image: vk::Image,
        usage: ImageUsage,
        format: vk::Format,
    ) {
        if let Some(last_usage) = state.insert(image, usage) {
            if last_usage == usage {
                return;
            }
            Self::cmd_image_2d_barrier(device, cmd_buffer, image, format, last_usage, usage);
        }
    }
}

impl GpuExecutor for V12Executor {
    type BType = V12Buffer;

    type I2dType = V12Image2d;

    type GFutType = V12Semaphore;

    type CFutType = V12Fence;

    type GPass = V12GraphicsPass;

    type E = V12ExecutorError;

    fn type_(&self) -> QueueType {
        self.type_
    }

    fn new_command_list(&mut self, name: &str) -> Result<(), Self::E> {
        let cmd_buffer = unsafe {
            self.device
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(V12ExecutorError::CommandBufferCreateError)?
                .pop()
                .ok_or(V12ExecutorError::CommandBufferCreateError(
                    vk::Result::ERROR_UNKNOWN,
                ))?
        };
        self.cmd_buffers.insert(name.to_string(), cmd_buffer);
        Ok(())
    }

    fn update_command_list(
        &mut self,
        name: &str,
        commands: Vec<GpuCommand<Self::BType, Self::I2dType, Self::GPass>>,
    ) -> Result<(), Self::E> {
        let Some(cmd_buffer) = self.cmd_buffers.get(name).cloned() else {
            return Err(V12ExecutorError::UnknownCommandBuffer(name.to_string()));
        };
        unsafe {
            self.device
                .device
                .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(V12ExecutorError::CommandBufferBeginError)?;
        }
        let mut img_state = HashMap::new();
        for command in commands {
            match command {
                GpuCommand::Image2dUsageHint { image, usage } => {
                    Self::update_image_usage(
                        &mut img_state,
                        &self.device,
                        cmd_buffer,
                        image.image,
                        usage,
                        image.format,
                    );
                }
                GpuCommand::CopyBufferToImage2d { src, dst } => {
                    Self::update_image_usage(
                        &mut img_state,
                        &self.device,
                        cmd_buffer,
                        dst.image,
                        ImageUsage::CopyDst,
                        dst.format,
                    );
                    unsafe {
                        self.device.device.cmd_copy_buffer_to_image(
                            cmd_buffer,
                            src.buffer,
                            dst.image,
                            image_usage_to_layout(ImageUsage::CopyDst, dst.format),
                            &[vk::BufferImageCopy::default()
                                .image_extent(dst.extent_3d())
                                .image_subresource(dst.subresource_layers())],
                        );
                    }
                }
                GpuCommand::BlitImage2d { src, dst } => {
                    Self::update_image_usage(
                        &mut img_state,
                        &self.device,
                        cmd_buffer,
                        src.image,
                        ImageUsage::CopySrc,
                        src.format,
                    );
                    Self::update_image_usage(
                        &mut img_state,
                        &self.device,
                        cmd_buffer,
                        dst.image,
                        ImageUsage::CopyDst,
                        dst.format,
                    );
                    unsafe {
                        self.device.device.cmd_blit_image(
                            cmd_buffer,
                            src.image,
                            image_usage_to_layout(ImageUsage::CopySrc, src.format),
                            dst.image,
                            image_usage_to_layout(ImageUsage::CopyDst, dst.format),
                            &[vk::ImageBlit::default()
                                .src_offsets(src.full_size_offset())
                                .src_subresource(src.subresource_layers())
                                .dst_offsets(dst.full_size_offset())
                                .dst_subresource(dst.subresource_layers())],
                            vk::Filter::NEAREST,
                        );
                    }
                }
                GpuCommand::RunGraphicsPass {
                    pass,
                    attachments,
                    commands,
                } => {
                    unsafe {
                        self.device.device.cmd_begin_render_pass(
                            cmd_buffer,
                            &vk::RenderPassBeginInfo::default()
                                .render_pass(pass.render_pass)
                                .framebuffer(attachments.framebuffer)
                                .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                                    width: attachments.res.width,
                                    height: attachments.res.height,
                                })),
                            vk::SubpassContents::INLINE,
                        );
                    }
                    for gpass_cmd in commands {
                        match gpass_cmd {
                            GraphicsPassCommand::BindSubpass { idx, sets } => unsafe {
                                self.device.device.cmd_bind_pipeline(
                                    cmd_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pass.pipelines[idx],
                                );
                                self.device.device.cmd_bind_descriptor_sets(
                                    cmd_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pass.pipeline_layouts[idx],
                                    0,
                                    &sets.iter().map(|s| s.set).collect::<Vec<_>>(),
                                    &[],
                                );
                            },
                            GraphicsPassCommand::Draw(count) => unsafe {
                                self.device.device.cmd_draw(cmd_buffer, count as _, 1, 0, 0);
                            },
                        }
                    }
                    unsafe {
                        self.device.device.cmd_end_render_pass(cmd_buffer);
                    }
                }
            }
        }
        unsafe {
            self.device
                .device
                .end_command_buffer(cmd_buffer)
                .map_err(V12ExecutorError::CommandBufferEndError)?;
        }
        Ok(())
    }

    fn run_command_lists(
        &self,
        lists: &[&str],
        wait_for: Vec<&Self::GFutType>,
        emit_gfuts: Vec<&Self::GFutType>,
        emit_cfut: Option<&Self::CFutType>,
    ) -> Result<(), Self::E> {
        let cmd_buffers: Vec<_> = lists
            .iter()
            .filter_map(|&n| self.cmd_buffers.get(n).cloned())
            .collect();
        let wait_semaphores: Vec<_> = wait_for.iter().map(|s| s.semaphore).collect();
        let emit_semaphores: Vec<_> = emit_gfuts.iter().map(|s| s.semaphore).collect();
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&emit_semaphores);

        unsafe {
            self.device
                .device
                .queue_submit(
                    self.queue,
                    &[submit_info],
                    emit_cfut.map(|f| f.fence).unwrap_or(vk::Fence::null()),
                )
                .map_err(V12ExecutorError::QueueSubmissionError)?
        }
        Ok(())
    }
}

impl Drop for V12Executor {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_command_pool(self.cmd_pool, None);
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
    ExecutorError(#[from] V12ExecutorError),
    #[error("'image' library related error: {0}")]
    ImageLibError(#[from] image::ImageError),
    #[error("Graphics Pass related error: {0}")]
    GraphicsPassError(#[from] V12GraphicsPassError),
}

pub struct V12Context {
    command_pool: vk::CommandPool,
    device: Arc<V12Device>,
}

impl GpuContext for V12Context {
    type AllocatorType = VkMemAllocator;

    type AllocationType = VkMemAllocation;

    type BType = V12Buffer;

    type I2dType = V12Image2d;

    type SwapchainType = V12Swapchain;

    type QType = V12Executor;

    type PSetType = V12PipelineSet;

    type PAttachType = V12GPassAttachments;

    type GPassType = V12GraphicsPass;

    type SemType = V12Semaphore;

    type FenType = V12Fence;

    type E = V12ContextError;

    fn new_buffer(
        &self,
        allocator: &mut Self::AllocatorType,
        gpu_local: bool,
        size: u64,
        name: &str,
        usage: BitFlags<BufferUsage>,
    ) -> Result<Self::BType, Self::E> {
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
    ) -> Result<Self::I2dType, Self::E> {
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

    fn new_swapchain(
        &self,
        mut usages: BitFlags<ImageUsage>,
    ) -> Result<Self::SwapchainType, Self::E> {
        usages |= ImageUsage::PipelineAttachment;
        usages |= ImageUsage::Present;
        let swapchain = V12Swapchain::new(self.device.clone(), usages)?;
        Ok(swapchain)
    }

    fn new_gpu_future(&self) -> Result<Self::SemType, Self::E> {
        let sem = V12Semaphore::new(self.device.clone())?;
        Ok(sem)
    }

    fn new_cpu_future(&self, signalled: bool) -> Result<Self::FenType, Self::E> {
        let fence = V12Fence::new(self.device.clone(), signalled)?;
        Ok(fence)
    }

    fn new_graphics_pass(
        &self,
        attachments: Vec<ImageFormat>,
        subpass_infos: Vec<SubpassInfo>,
        max_sets: usize,
    ) -> Result<Self::GPassType, Self::E> {
        let g_pass =
            V12GraphicsPass::new(self.device.clone(), attachments, subpass_infos, max_sets)?;
        Ok(g_pass)
    }

    fn get_queue(&mut self) -> Result<Self::QType, Self::E> {
        Ok(V12Executor::new(self.device.clone(), QueueType::Graphics)?)
    }
}

impl Drop for V12Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
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
            .application_name(c"Cougher App")
            .application_version(1)
            .engine_name(c"Cougher Vulkan 1.2")
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
        let queues = HashMap::from([(QueueType::Graphics, (gpu.g_queue_family.0 as u32, g_queue))]);

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
                queues,
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

pub struct V12Device {
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queues: HashMap<QueueType, (u32, vk::Queue)>,
    pub(crate) swapchain_device: khr::swapchain::Device,
    pub(crate) device: ash::Device,
    pub(crate) loader: V12ApiLoader,
}

impl Drop for V12Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
