use std::sync::{Arc, Mutex};

use ash::vk::{self, Handle};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

use crate::vk_wrap::device::{AllocError, Device};

fn aspect_flags(depth: bool, stencil: bool) -> vk::ImageAspectFlags {
    if depth {
        if stencil {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

fn format_is_depth(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM
        | vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT
        | vk::Format::D32_SFLOAT_S8_UINT
        | vk::Format::X8_D24_UNORM_PACK32 => true,
        _ => false,
    }
}

fn format_has_stencil(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImageErrorVk {
    #[error("Error creating Vulkan Sampler: {0}")]
    SamplerCreateError(vk::Result),
    #[error("Error creating Vulkan Image: {0}")]
    ImageCreateError(vk::Result),
    #[error("Error creating Vulkan Image View: {0}")]
    ImageViewCreateError(vk::Result),
    #[error("Error allocation memory for  Vulkan Image: {0}")]
    AllocationError(AllocError),
    #[error("Error binding memeory to Image: {0}")]
    MemoryBindError(vk::Result),
}

pub struct Image2d {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) memory: Option<Allocation>,
    pub(crate) location: MemoryLocation,
    pub(crate) extent: vk::Extent2D,
    pub(crate) format: vk::Format,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) allocator: Option<Arc<Mutex<Allocator>>>,
    pub(crate) device: Arc<Device>,
}

impl Image2d {
    pub fn new(
        device: &Arc<Device>,
        allocator: &Arc<Mutex<Allocator>>,
        location: MemoryLocation,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Image2d, ImageErrorVk> {
        let extent_3d = vk::Extent3D::default()
            .width(extent.width)
            .height(extent.height)
            .depth(1);
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent_3d)
            .array_layers(1)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .format(format)
            .usage(usage);
        let image = unsafe {
            device
                .device
                .create_image(&create_info, None)
                .map_err(ImageErrorVk::ImageCreateError)?
        };
        let mem_req = unsafe { device.device.get_image_memory_requirements(image) };
        let memory = allocator
            .lock()
            .map_err(|e| ImageErrorVk::AllocationError(AllocError::LockError(format!("{e}"))))?
            .allocate(&AllocationCreateDesc {
                name: &format!("{:x}", image.as_raw()),
                requirements: mem_req,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(AllocError::LibError)
            .map_err(ImageErrorVk::AllocationError)?;
        unsafe {
            device
                .device
                .bind_image_memory(image, memory.memory(), memory.offset())
                .map_err(ImageErrorVk::MemoryBindError)?;
        }
        let view = if usage.contains(vk::ImageUsageFlags::SAMPLED)
            || usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            || usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            unsafe {
                device
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .format(format)
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D),
                        None,
                    )
                    .map_err(ImageErrorVk::ImageViewCreateError)?
            }
        } else {
            vk::ImageView::null()
        };
        Ok(Self {
            image,
            memory: Some(memory),
            view,
            location,
            extent,
            format,
            usage,
            allocator: Some(allocator.clone()),
            device: device.clone(),
        })
    }

    pub fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        Self::subresource_layers_fmt(self.format)
    }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        Self::subresource_range_fmt(self.format)
    }

    pub fn subresource_layers_stc(depth: bool, stencil: bool) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(aspect_flags(depth, stencil))
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0)
    }

    pub fn subresource_range_stc(depth: bool, stencil: bool) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_flags(depth, stencil))
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(1)
    }

    pub fn subresource_layers_fmt(fmt: vk::Format) -> vk::ImageSubresourceLayers {
        let depth = format_is_depth(fmt);
        let stencil = format_has_stencil(fmt);
        Self::subresource_layers_stc(depth, stencil)
    }

    pub fn subresource_range_fmt(fmt: vk::Format) -> vk::ImageSubresourceRange {
        let depth = format_is_depth(fmt);
        let stencil = format_has_stencil(fmt);
        Self::subresource_range_stc(depth, stencil)
    }
}

impl Drop for Image2d {
    fn drop(&mut self) {
        unsafe {
            let Some(memory) = self.memory.take() else {
                return;
            };
            let Some(allocator) = self.allocator.take() else {
                return;
            };
            self.device.device.destroy_image(self.image, None);
            match allocator.lock() {
                Ok(mut altr) => {
                    if let Err(e) = altr.free(memory) {
                        eprintln!(
                            "error freeing memory of image {:x}: {e}",
                            self.image.as_raw()
                        )
                    }
                }
                Err(e) => eprintln!(
                    "error getting allocator mutex lock for freeing memory of image {:x}: {e}",
                    self.image.as_raw()
                ),
            };
        }
    }
}

pub struct Sampler {
    sampler: vk::Sampler,
    device: Arc<Device>,
}

impl Sampler {
    pub fn new(device: &Arc<Device>) -> Result<Self, ImageErrorVk> {
        let sampler = unsafe {
            device
                .device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .map_err(ImageErrorVk::SamplerCreateError)?
        };
        Ok(Sampler {
            sampler,
            device: device.clone(),
        })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.sampler, None);
        }
    }
}
