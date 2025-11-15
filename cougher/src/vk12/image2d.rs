use std::mem::ManuallyDrop;

use ash::vk::{self, Handle};
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

#[derive(Debug, thiserror::Error)]
pub enum Image2dError {
    #[error("Error creating Vulkan Image: {0}")]
    ImageCreateError(vk::Result),
    #[error("Error allocation memory for  Vulkan Image: {0}")]
    AllocationError(AllocationError),
}

pub struct Image2d {
    pub(crate) image: vk::Image,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent2D,
    pub(crate) allocation: ManuallyDrop<Allocation>,
    needs_cleanup: bool,
}

impl Image2d {
    pub fn image_subresource_layers(depth: bool, stencil: bool) -> vk::ImageSubresourceLayers {
        let aspect_mask = if depth {
            if stencil {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };
        vk::ImageSubresourceLayers::default()
            .aspect_mask(aspect_mask)
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0)
    }

    pub fn is_depth(&self) -> (bool, bool) {
        match self.format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
                (true, false)
            }
            vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => (true, true),
            _ => (false, false),
        }
    }

    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        location: MemoryLocation,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self, Image2dError> {
        let create_info = vk::ImageCreateInfo::default()
            .extent(
                vk::Extent3D::default()
                    .width(extent.width)
                    .height(extent.height)
                    .depth(1),
            )
            .format(format);
        let image = unsafe {
            device
                .create_image(&create_info, None)
                .map_err(Image2dError::ImageCreateError)?
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
            .map(ManuallyDrop::new)
            .map_err(Image2dError::AllocationError)?;

        Ok(Self {
            image,
            format,
            extent,
            allocation,
            needs_cleanup: true,
        })
    }

    pub fn cleanup(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if self.needs_cleanup {
            unsafe {
                device.destroy_image(self.image, None);
                let _ = allocator
                    .free(ManuallyDrop::take(&mut self.allocation))
                    .inspect_err(|e| eprintln!("warning: error cleaning up gpu allocation: {e}"));
            }
        }
    }
}
