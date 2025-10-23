use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;
use enumflags2::BitFlags;
use gpu_allocator::{
    AllocationError, AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocation, Allocator, AllocatorCreateDesc},
};

use crate::traits::{BufferUsage, ImageFormat, MemAllocation, MemAllocator, Resolution2d};

pub struct VkMemAllocator {
    pub(crate) allocator: Arc<Mutex<Allocator>>,
}

impl VkMemAllocator {
    pub fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, AllocationError> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: AllocationSizes::default(),
        })?;
        Ok(VkMemAllocator {
            allocator: Arc::new(Mutex::new(allocator)),
        })
    }
}

impl MemAllocator for VkMemAllocator {}

pub struct VkMemAllocation {
    pub(crate) allocation: ManuallyDrop<Allocation>,
    pub(crate) allocator: Arc<Mutex<Allocator>>,
    pub(crate) is_gpu_local: bool,
}

impl MemAllocation for VkMemAllocation {
    type AllocatorType = VkMemAllocator;

    fn is_gpu_local(&self) -> bool {
        self.is_gpu_local
    }
}

impl Drop for VkMemAllocation {
    fn drop(&mut self) {
        unsafe {
            let Ok(mut allocator_mut) = self.allocator.lock() else {
                return;
            };
            allocator_mut
                .free(ManuallyDrop::take(&mut self.allocation))
                .inspect_err(|e| eprintln!("error feeing memory: {e}"))
                .ok();
        }
    }
}

pub fn format_to_vk(format: ImageFormat) -> vk::Format {
    match format {
        ImageFormat::R32 => vk::Format::R32_SFLOAT,
        ImageFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        ImageFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        ImageFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        ImageFormat::D24S8 => vk::Format::D24_UNORM_S8_UINT,
        ImageFormat::D32 => vk::Format::D32_SFLOAT,
        ImageFormat::D32S8 => vk::Format::D32_SFLOAT_S8_UINT,
    }
}

pub fn valid_formats() -> Vec<vk::Format> {
    vec![]
}

pub fn vk_to_format(format: vk::Format) -> Option<ImageFormat> {
    match format {
        vk::Format::R32_SFLOAT => Some(ImageFormat::R32),
        vk::Format::R8G8B8A8_SRGB => Some(ImageFormat::Rgba8Srgb),
        vk::Format::B8G8R8A8_SRGB => Some(ImageFormat::Bgra8Srgb),
        vk::Format::R8G8B8A8_UNORM => Some(ImageFormat::Rgba8),
        vk::Format::D24_UNORM_S8_UINT => Some(ImageFormat::D24S8),
        vk::Format::D32_SFLOAT => Some(ImageFormat::D32),
        vk::Format::D32_SFLOAT_S8_UINT => Some(ImageFormat::D32S8),
        _ => None,
    }
}

pub fn buffer_usage_to_vk(usages: BitFlags<BufferUsage>) -> vk::BufferUsageFlags {
    let mut vk_flags = vk::BufferUsageFlags::default();
    for usage in usages {
        match usage {
            BufferUsage::None => {}
            BufferUsage::Uniform => vk_flags |= vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferUsage::Storage => vk_flags |= vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferUsage::TransferSrc => vk_flags |= vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsage::TransferDst => vk_flags |= vk::BufferUsageFlags::TRANSFER_DST,
        }
    }
    vk_flags
}

pub fn res_to_extent_2d(res: Resolution2d) -> vk::Extent2D {
    vk::Extent2D {
        width: res.width,
        height: res.height,
    }
}

pub fn res_to_extent_3d(res: Resolution2d) -> vk::Extent3D {
    vk::Extent3D {
        width: res.width,
        height: res.height,
        depth: 1,
    }
}

pub fn format_to_aspect_mask(format: ImageFormat) -> vk::ImageAspectFlags {
    if format.is_depth() {
        let mut aspect = vk::ImageAspectFlags::DEPTH;
        if format.has_stencil() {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        aspect
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

pub fn image_2d_subresource_range(format: ImageFormat) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(format_to_aspect_mask(format))
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1)
}

pub fn image_2d_subresource_layers(format: ImageFormat) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(format_to_aspect_mask(format))
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0)
}
