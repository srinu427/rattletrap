use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use thiserror::Error;

use crate::wrappers::{
    context::Context,
    gpu_allocation::{GpuAllocation, GpuAllocationError},
    logical_device::LogicalDevice,
};

#[derive(Debug, Error)]
pub enum Image2dError {
    #[error("Error creating image: {0}")]
    CreateError(vk::Result),
    #[error("GPU Allocation error: {0}")]
    AllocationError(#[from] GpuAllocationError),
    #[error("Error binding memory to image: {0}")]
    MemoryBindError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Image {
    #[get_copy = "pub"]
    image: vk::Image,
    #[get_copy = "pub"]
    type_: vk::ImageType,
    #[get_copy = "pub"]
    format: vk::Format,
    #[get_copy = "pub"]
    extent: vk::Extent3D,
    #[get_copy = "pub"]
    mip_levels: u32,
    #[get_copy = "pub"]
    array_layers: u32,
    #[get_copy = "pub"]
    usage: vk::ImageUsageFlags,
    #[get = "pub"]
    allocation: Option<GpuAllocation>,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Image {
    pub fn new(
        context: &Context,
        type_: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        mip_levels: u32,
        array_layers: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, Image2dError> {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(type_)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage);

        let image = unsafe {
            context
                .logical_device()
                .device
                .create_image(&image_create_info, None)
                .map_err(Image2dError::CreateError)?
        };

        Ok(Self {
            image,
            type_,
            allocation: None,
            device: context.logical_device().clone(),
            extent,
            format,
            usage,
            mip_levels,
            array_layers,
        })
    }

    pub fn new_2d(
        context: &Context,
        format: vk::Format,
        extent: vk::Extent2D,
        mip_levels: u32,
        array_layers: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, Image2dError> {
        Self::new(
            context,
            vk::ImageType::TYPE_2D,
            format,
            vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels,
            array_layers,
            usage,
        )
    }

    pub fn allocate_memory(
        &mut self,
        allocator: Arc<Mutex<Allocator>>,
        gpu_only: bool,
    ) -> Result<(), Image2dError> {
        let requirements = unsafe { self.device.device.get_image_memory_requirements(self.image) };
        let mem_location = if gpu_only {
            gpu_allocator::MemoryLocation::GpuOnly
        } else {
            gpu_allocator::MemoryLocation::CpuToGpu
        };
        let allocation = GpuAllocation::new(
            allocator,
            &format!("image_{:?}", self.image),
            requirements,
            false,
            mem_location,
        )?;
        unsafe {
            self.device
                .device
                .bind_image_memory(
                    self.image,
                    allocation.allocation().memory(),
                    allocation.allocation().offset(),
                )
                .map_err(Image2dError::MemoryBindError)?;
        }
        Ok(())
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image(self.image, None);
        }
    }
}
