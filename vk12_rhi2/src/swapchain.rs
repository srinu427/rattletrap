use std::sync::Arc;

use ash::vk;

use crate::device::DeviceDropper;

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    fence: vk::Fence,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    width: u32,
    height: u32,
    device_dropper: Arc<DeviceDropper>,
}

impl Swapchain {
    pub fn new(device: &Arc<DeviceDropper>) -> Result<Self, String> {
        todo!()
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .swapchain_device
                .destroy_swapchain(self.handle, None);
        }
    }
}
