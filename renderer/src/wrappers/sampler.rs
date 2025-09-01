use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    #[error("Sampler creation error: {0}")]
    CreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Sampler {
    #[get_copy = "pub"]
    sampler: vk::Sampler,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Sampler {
    pub fn new_nearest(device: Arc<LogicalDevice>) -> Result<Self, SamplerError> {
        let create_info = vk::SamplerCreateInfo::default();

        let sampler = unsafe {
            device
                .device()
                .create_sampler(&create_info, None)
                .map_err(SamplerError::CreateError)?
        };

        Ok(Self { sampler, device })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_sampler(self.sampler, None);
        }
    }
}
