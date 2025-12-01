use std::sync::Arc;

use ash::vk;

use crate::vk_wrap::device::Device;

#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error("Error creating Vulkan Semaphore: {0}")]
    SemaphoreCreateError(vk::Result),
    #[error("Error creating Vulkan Fence: {0}")]
    FenceCreateError(vk::Result),
    #[error("Error waiting for Vulkan Fences: {0}")]
    FenceWaitError(vk::Result),
    #[error("Error resetting for Vulkan Fences: {0}")]
    FenceResetError(vk::Result),
}

pub struct SemStageInfo<'a> {
    pub(crate) sem: &'a Semaphore,
    pub(crate) stage: vk::PipelineStageFlags,
}

pub struct Semaphore {
    pub(crate) sem: vk::Semaphore,
    pub(crate) device: Arc<Device>,
}

impl Semaphore {
    pub fn new(device: &Arc<Device>) -> Result<Self, SyncError> {
        let sem = unsafe {
            device
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(SyncError::SemaphoreCreateError)?
        };
        Ok(Self {
            sem,
            device: device.clone(),
        })
    }

    pub fn stage_info(&self, stage: vk::PipelineStageFlags) -> SemStageInfo<'_> {
        SemStageInfo { sem: self, stage }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_semaphore(self.sem, None);
        }
    }
}

pub struct Fence {
    pub(crate) fence: vk::Fence,
    pub(crate) device: Arc<Device>,
}

impl Fence {
    pub fn new(device: &Arc<Device>, signalled: bool) -> Result<Self, SyncError> {
        let flags = if signalled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let fence = unsafe {
            device
                .device
                .create_fence(&vk::FenceCreateInfo::default().flags(flags), None)
                .map_err(SyncError::FenceCreateError)?
        };
        Ok(Self {
            fence,
            device: device.clone(),
        })
    }

    pub fn wait(&self, timeout: Option<u64>) -> Result<(), SyncError> {
        unsafe {
            self.device
                .device
                .wait_for_fences(&[self.fence], true, timeout.unwrap_or(u64::MAX))
                .map_err(SyncError::FenceWaitError)
        }
    }

    pub fn reset(&self) -> Result<(), SyncError> {
        unsafe {
            self.device
                .device
                .reset_fences(&[self.fence])
                .map_err(SyncError::FenceResetError)
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
) -> Result<(), SyncError> {
    if fences.is_empty() {
        return Ok(());
    }
    let fences_vk: Vec<_> = fences.iter().map(|f| f.fence).collect();
    unsafe {
        device
            .wait_for_fences(&fences_vk, true, timeout.unwrap_or(u64::MAX))
            .map_err(SyncError::FenceWaitError)
    }
}

pub fn reset_fences(device: &ash::Device, fences: &[&Fence]) -> Result<(), SyncError> {
    if fences.is_empty() {
        return Ok(());
    }
    let fences_vk: Vec<_> = fences.iter().map(|f| f.fence).collect();
    unsafe {
        device
            .reset_fences(&fences_vk)
            .map_err(SyncError::FenceResetError)
    }
}
