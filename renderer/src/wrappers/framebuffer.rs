use std::sync::Arc;

use ash::vk;

use crate::wrappers::{
    image_view::ImageView, logical_device::LogicalDevice, render_pass::RenderPass,
};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Framebuffer {
    #[get_copy = "pub"]
    framebuffer: vk::Framebuffer,
    attachments: Vec<Arc<ImageView>>,
    #[get_copy = "pub"]
    extent: vk::Extent2D,
    #[get = "pub"]
    render_pass: Arc<RenderPass>,
}

impl Framebuffer {
    pub fn new(
        render_pass: Arc<RenderPass>,
        attachments: Vec<Arc<ImageView>>,
        extent: vk::Extent2D,
        layers: u32,
    ) -> Result<Self, vk::Result> {
        let vk_attachments = attachments
            .iter()
            .map(|view| view.image_view())
            .collect::<Vec<_>>();
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass.render_pass())
            .attachments(&vk_attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(layers);

        let framebuffer = unsafe {
            render_pass
                .device()
                .device()
                .create_framebuffer(&create_info, None)?
        };

        Ok(Self {
            framebuffer,
            attachments,
            render_pass,
            extent,
        })
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
