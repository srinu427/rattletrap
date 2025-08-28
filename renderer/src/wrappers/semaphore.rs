use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Semaphore {
    #[get_copy = "pub"]
    semaphore: vk::Semaphore,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Semaphore {
    pub fn new(device: Arc<LogicalDevice>) -> Result<Self, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let semaphore = unsafe { device.device().create_semaphore(&create_info, None)? };

        Ok(Self { semaphore, device })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_semaphore(self.semaphore, None);
        }
    }
}
