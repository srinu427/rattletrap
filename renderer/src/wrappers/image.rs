use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

pub struct Image {
    pub(crate) image: vk::Image,
    pub(crate) device: Arc<LogicalDevice>,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image(self.image, None);
        }
    }
}
