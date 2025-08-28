use std::sync::Arc;

use ash::vk;

use crate::wrappers::{image_view::ImageView, logical_device::LogicalDevice, render_pass::RenderPass};

pub struct Framebuffer {
    pub framebuffer: vk::Framebuffer,
    attachments: Vec<Arc<ImageView>>,
    render_pass: Arc<RenderPass>,
}

impl Framebuffer {
    pub fn new(
        device: Arc<LogicalDevice>,
        render_pass: Arc<RenderPass>,
        attachments: Vec<Arc<ImageView>>,
        width: u32,
        height: u32,
        layers: u32,
    ) -> Result<Self, vk::Result> {
        let vk_attachments = attachments
            .iter()
            .map(|view| view.image_view())
            .collect::<Vec<_>>();
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass.render_pass())
            .attachments(&vk_attachments)
            .width(width)
            .height(height)
            .layers(layers);

        let framebuffer = unsafe { device.device().create_framebuffer(&create_info, None)? };

        Ok(Self { framebuffer, attachments, render_pass } )
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.render_pass
                .device()
                .device()
                .destroy_framebuffer(self.framebuffer, None);
        }
    }
}