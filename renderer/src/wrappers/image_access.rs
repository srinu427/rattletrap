use ash::vk;

#[derive(Debug, Clone, Copy)]
pub enum ImageAccess {
    None,
    TransferRead,
    TransferWrite,
    ShaderRead,
    ColorAttachment,
    DepthAttachment,
    Present,
}

impl ImageAccess {
    pub fn to_access_flags(&self) -> vk::AccessFlags {
        match self {
            ImageAccess::None => vk::AccessFlags::empty(),
            ImageAccess::TransferRead => vk::AccessFlags::TRANSFER_READ,
            ImageAccess::TransferWrite => vk::AccessFlags::TRANSFER_WRITE,
            ImageAccess::ShaderRead => vk::AccessFlags::SHADER_READ,
            ImageAccess::ColorAttachment => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ImageAccess::DepthAttachment => {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            ImageAccess::Present => vk::AccessFlags::MEMORY_READ,
        }
    }

    pub fn to_usage_flags(&self) -> vk::ImageUsageFlags {
        match self {
            ImageAccess::None => vk::ImageUsageFlags::empty(),
            ImageAccess::TransferRead => vk::ImageUsageFlags::TRANSFER_SRC,
            ImageAccess::TransferWrite => vk::ImageUsageFlags::TRANSFER_DST,
            ImageAccess::ShaderRead => vk::ImageUsageFlags::SAMPLED,
            ImageAccess::ColorAttachment => {
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE
            }
            ImageAccess::DepthAttachment => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageAccess::Present => vk::ImageUsageFlags::empty(),
        }
    }

    pub fn get_image_layout(&self) -> vk::ImageLayout {
        match self {
            ImageAccess::None => vk::ImageLayout::UNDEFINED,
            ImageAccess::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageAccess::TransferWrite => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageAccess::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageAccess::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ImageAccess::DepthAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ImageAccess::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    pub fn get_pipeline_stage(&self) -> vk::PipelineStageFlags {
        match self {
            ImageAccess::None => vk::PipelineStageFlags::TOP_OF_PIPE,
            ImageAccess::TransferRead => vk::PipelineStageFlags::TRANSFER,
            ImageAccess::TransferWrite => vk::PipelineStageFlags::TRANSFER,
            ImageAccess::ShaderRead => vk::PipelineStageFlags::FRAGMENT_SHADER,
            ImageAccess::ColorAttachment => vk::PipelineStageFlags::ALL_GRAPHICS,
            ImageAccess::DepthAttachment => vk::PipelineStageFlags::ALL_GRAPHICS,
            ImageAccess::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        }
    }
}
