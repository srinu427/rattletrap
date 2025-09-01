use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(Debug, thiserror::Error)]
pub enum SemaphoreError {
    #[error("Semaphore creation error: {0}")]
    CreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Semaphore {
    #[get_copy = "pub"]
    semaphore: vk::Semaphore,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Semaphore {
    pub fn new(device: Arc<LogicalDevice>) -> Result<Self, SemaphoreError> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let semaphore = unsafe {
            device
                .device()
                .create_semaphore(&create_info, None)
                .map_err(SemaphoreError::CreateError)?
        };

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
