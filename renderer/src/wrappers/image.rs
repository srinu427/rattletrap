use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use thiserror::Error;

use crate::wrappers::{
    gpu_allocation::{GpuAllocation, GpuAllocationError},
    logical_device::LogicalDevice,
};

fn is_depth_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM
            | vk::Format::X8_D24_UNORM_PACK32
            | vk::Format::D32_SFLOAT
            | vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

fn has_stencil_aspect(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

fn get_aspect_from_format(format: vk::Format) -> vk::ImageAspectFlags {
    if is_depth_format(format) {
        if has_stencil_aspect(format) {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageAccess {
    Undefined,
    Attachment,
    ShaderRead,
    TransferSrc,
    TransferDst,
    BlitSrc,
    BlitDst,
    Present,
}

impl ImageAccess {
    pub fn to_layout(&self, _format: vk::Format) -> vk::ImageLayout {
        match self {
            ImageAccess::Undefined => vk::ImageLayout::UNDEFINED,
            ImageAccess::Attachment => vk::ImageLayout::ATTACHMENT_OPTIMAL,
            ImageAccess::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageAccess::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageAccess::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageAccess::BlitSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageAccess::BlitDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageAccess::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    pub fn to_stage_flags(&self, format: vk::Format) -> vk::PipelineStageFlags2 {
        match self {
            ImageAccess::Undefined => vk::PipelineStageFlags2::ALL_COMMANDS,
            ImageAccess::Attachment => {
                if is_depth_format(format) {
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                } else {
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                }
            }
            ImageAccess::ShaderRead => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ImageAccess::TransferSrc | ImageAccess::TransferDst => {
                vk::PipelineStageFlags2::TRANSFER
            }
            ImageAccess::BlitSrc | ImageAccess::BlitDst => vk::PipelineStageFlags2::BLIT,
            ImageAccess::Present => vk::PipelineStageFlags2::ALL_COMMANDS,
        }
    }

    pub fn to_access_flags(&self, format: vk::Format) -> vk::AccessFlags2 {
        match self {
            ImageAccess::Undefined => vk::AccessFlags2::NONE,
            ImageAccess::Attachment => {
                if is_depth_format(format) {
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                } else {
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                }
            }
            ImageAccess::ShaderRead => vk::AccessFlags2::SHADER_READ,
            ImageAccess::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            ImageAccess::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
            ImageAccess::BlitSrc => vk::AccessFlags2::TRANSFER_READ,
            ImageAccess::BlitDst => vk::AccessFlags2::TRANSFER_WRITE,
            ImageAccess::Present => vk::AccessFlags2::NONE,
        }
    }

    pub fn to_usage_flags(&self, format: vk::Format) -> vk::ImageUsageFlags {
        match self {
            ImageAccess::Undefined => vk::ImageUsageFlags::empty(),
            ImageAccess::Attachment => {
                if is_depth_format(format) {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                }
            }
            ImageAccess::ShaderRead => vk::ImageUsageFlags::SAMPLED,
            ImageAccess::TransferSrc => vk::ImageUsageFlags::TRANSFER_SRC,
            ImageAccess::TransferDst => vk::ImageUsageFlags::TRANSFER_DST,
            ImageAccess::BlitSrc => vk::ImageUsageFlags::TRANSFER_SRC,
            ImageAccess::BlitDst => vk::ImageUsageFlags::TRANSFER_DST,
            ImageAccess::Present => vk::ImageUsageFlags::empty(),
        }
    }

    pub fn to_usage_flags_vec(usages: &[ImageAccess], format: vk::Format) -> vk::ImageUsageFlags {
        let mut flags = vk::ImageUsageFlags::empty();
        for usage in usages {
            flags |= usage.to_usage_flags(format);
        }
        flags
    }
}

#[derive(Debug, Error)]
pub enum ImageError {
    #[error("Image creation error: {0}")]
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
    #[get = "pub"]
    usage: Vec<ImageAccess>,
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
        usage: Vec<ImageAccess>,
    ) -> Result<Self, ImageError> {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(type_)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(ImageAccess::to_usage_flags_vec(&usage, format));

        let image = unsafe {
            device
                .device()
                .create_image(&image_create_info, None)
                .map_err(ImageError::CreateError)?
        };

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
        usage: Vec<ImageAccess>,
    ) -> Result<Self, ImageError> {
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
            1,
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
            true,
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

    pub(crate) fn from_swapchain_image(
        device: Arc<LogicalDevice>,
        image: vk::Image,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        Self {
            image,
            need_delte: false,
            type_: vk::ImageType::TYPE_2D,
            allocation: None,
            device,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            format,
            usage: vec![],
            mip_levels: 1,
            array_layers: 1,
        }
    }

    pub fn full_subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(get_aspect_from_format(self.format))
            .base_mip_level(0)
            .level_count(self.mip_levels)
            .base_array_layer(0)
            .layer_count(self.array_layers)
    }

    pub fn all_subresource_layers(&self, mip_level: u32) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(get_aspect_from_format(self.format))
            .mip_level(mip_level)
            .base_array_layer(0)
            .layer_count(self.array_layers)
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
