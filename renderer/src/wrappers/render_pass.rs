use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(getset::Getters, getset::CopyGetters)]
pub struct RenderPass {
    #[get_copy = "pub"]
    render_pass: vk::RenderPass,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl RenderPass {
    pub fn new(
        device: Arc<LogicalDevice>,
        create_info: &vk::RenderPassCreateInfo2,
    ) -> Result<Self, vk::Result> {
        let render_pass = unsafe { device.device().create_render_pass2(create_info, None)? };

        Ok(Self {
            render_pass,
            device,
        })
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}
