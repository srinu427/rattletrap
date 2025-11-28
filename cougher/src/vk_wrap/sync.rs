use std::sync::Arc;

use ash::vk;

use crate::vk_wrap::device::Device;

#[derive(Debug, thiserror::Error)]
pub enum FenceError {
    #[error("Error creating Vulkan Fence: {0}")]
    CreateError(vk::Result),
    #[error("Error waiting for Vulkan Fences: {0}")]
    WaitError(vk::Result),
    #[error("Error resetting for Vulkan Fences: {0}")]
    ResetError(vk::Result),
}

pub struct Fence {
    pub(crate) fence: vk::Fence,
    pub(crate) device: Arc<Device>,
}

impl Fence {
    pub fn new(device: &Arc<Device>) -> Result<Self, FenceError> {
        let fence = unsafe {
            device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(FenceError::CreateError)?
        };
        Ok(Self {
            fence,
            device: device.clone(),
        })
    }

    pub fn wait(&self, timeout: Option<u64>) -> Result<(), FenceError> {
        unsafe {
            self.device
                .device
                .wait_for_fences(&[self.fence], true, timeout.unwrap_or(u64::MAX))
                .map_err(FenceError::WaitError)
        }
    }

    pub fn reset(&self) -> Result<(), FenceError> {
        unsafe {
            self.device
                .device
                .reset_fences(&[self.fence])
                .map_err(FenceError::ResetError)
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_fence(self.fence, None);
        }
    }
}

pub fn wait_for_fences(
    device: &ash::Device,
    fences: &[&Fence],
    timeout: Option<u64>,
) -> Result<(), FenceError> {
    let fences_vk: Vec<_> = fences.iter().map(|f| f.fence).collect();
    unsafe {
        device
            .wait_for_fences(&fences_vk, true, timeout.unwrap_or(u64::MAX))
            .map_err(FenceError::WaitError)
    }
}

pub fn reset_fences(device: &ash::Device, fences: &[&Fence]) -> Result<(), FenceError> {
    let fences_vk: Vec<_> = fences.iter().map(|f| f.fence).collect();
    unsafe {
        device
            .reset_fences(&fences_vk)
            .map_err(FenceError::ResetError)
    }
}
