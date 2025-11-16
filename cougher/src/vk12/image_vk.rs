use ash::vk::{self, Handle};
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

use crate::make_init_struct_copy;

make_init_struct_copy!(
    InitImage,
    vk::Image,
    self,
    self.device.destroy_image(self.inner, None)
);

#[derive(Debug, thiserror::Error)]
pub enum ImageErrorVk {
    #[error("Error creating Vulkan Image: {0}")]
    ImageCreateError(vk::Result),
    #[error("Error allocation memory for  Vulkan Image: {0}")]
    AllocationError(AllocationError),
}

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

pub fn image_subresource_layers_2d(depth: bool, stencil: bool) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(aspect_flags(depth, stencil))
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0)
}

pub fn image_subresource_range_2d(depth: bool, stencil: bool) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_flags(depth, stencil))
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1)
}

pub fn new_image_2d<'a>(
    device: &'a ash::Device,
    allocator: &'_ mut Allocator,
    location: MemoryLocation,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<(InitImage<'a>, Allocation), ImageErrorVk> {
    let extent = vk::Extent3D::default()
        .width(extent.width)
        .height(extent.height)
        .depth(1);
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(
            vk::Extent3D::default()
                .width(extent.width)
                .height(extent.height)
                .depth(1),
        )
        .array_layers(1)
        .mip_levels(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .format(format)
        .usage(usage);
    let image = unsafe {
        device
            .create_image(&create_info, None)
            .map_err(ImageErrorVk::ImageCreateError)?
    };
    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: &format!("{:x}", image.as_raw()),
            requirements: mem_req,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(ImageErrorVk::AllocationError)?;

    let init_i2d = InitImage {
        drop: true,
        inner: image,
        device,
    };
    Ok((init_i2d, allocation))
}
