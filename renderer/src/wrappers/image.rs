use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use thiserror::Error;

use crate::wrappers::{
    gpu_allocation::{GpuAllocation, GpuAllocationError},
    logical_device::LogicalDevice,
};

#[derive(Debug, Error)]
pub enum ImageError {
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
    need_delte: bool,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Image {
    pub fn new(
        device: Arc<LogicalDevice>,
        type_: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        mip_levels: u32,
        array_layers: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, vk::Result> {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(type_)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage);

        let image = unsafe { device.device().create_image(&image_create_info, None)? };

        Ok(Self {
            image,
            need_delte: true,
            type_,
            allocation: None,
            device,
            extent,
            format,
            usage,
            mip_levels,
            array_layers,
        })
    }

    pub fn new_2d(
        device: Arc<LogicalDevice>,
        format: vk::Format,
        extent: vk::Extent2D,
        mip_levels: u32,
        array_layers: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, vk::Result> {
        Self::new(
            device,
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
    ) -> Result<(), ImageError> {
        let requirements = unsafe {
            self.device
                .device()
                .get_image_memory_requirements(self.image)
        };
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
                .device()
                .bind_image_memory(
                    self.image,
                    allocation.allocation().memory(),
                    allocation.allocation().offset(),
                )
                .map_err(ImageError::MemoryBindError)?;
        }
        Ok(())
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            if self.need_delte {
                self.device.device().destroy_image(self.image, None);
            }
        }
    }
}
