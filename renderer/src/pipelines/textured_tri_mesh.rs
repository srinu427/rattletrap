use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::wrappers::{logical_device::LogicalDevice, render_pass::RenderPass};

#[derive(Debug, Error)]
pub enum TexturedTriMeshInitError {
    #[error("Render pass creation error: {0}")]
    RenderPassCreateError(vk::Result),
}

pub struct TexturedTriMesh {
    render_pass: Arc<RenderPass>,
}

impl TexturedTriMesh {
    pub fn new(device: Arc<LogicalDevice>) -> Result<Self, TexturedTriMeshInitError> {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .samples(vk::SampleCountFlags::TYPE_1),
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .samples(vk::SampleCountFlags::TYPE_1),
        ];
        let color_attach_refs = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth_attach_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attach_refs)
            .depth_stencil_attachment(&depth_attach_ref)];
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .subpasses(&subpasses)
            .attachments(&attachments);
        let render_pass = RenderPass::new(device, &render_pass_create_info)
            .map(Arc::new)
            .map_err(TexturedTriMeshInitError::RenderPassCreateError)?;

        Ok(Self { render_pass })
    }
}
