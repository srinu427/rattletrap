use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::wrappers::{buffer::Buffer, logical_device::LogicalDevice};

#[derive(Debug, Error)]
pub enum FenceError {
    #[error("Fence creation error: {0}")]
    CreateError(vk::Result),
    #[error("Fence wait error: {0}")]
    WaitError(vk::Result),
    #[error("Fence reset error: {0}")]
    ResetError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Fence {
    preserve_buffers: Vec<Buffer>,
    #[get_copy = "pub"]
    fence: vk::Fence,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Fence {
    pub fn new(device: Arc<LogicalDevice>, signaled: bool) -> Result<Self, FenceError> {
        let create_info = vk::FenceCreateInfo::default().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });

        let fence = unsafe {
            device
                .device()
                .create_fence(&create_info, None)
                .map_err(FenceError::CreateError)?
        };

        Ok(Self { preserve_buffers: vec![], fence, device })
    }

    pub fn preserve_buffer(&mut self, buffer: Buffer) {
        self.preserve_buffers.push(buffer);
    }

    pub fn flush_buffers(&mut self) {
        self.preserve_buffers.clear();
    }

    pub fn wait(&self, timeout: u64) -> Result<(), FenceError> {
        unsafe {
            self.device
                .device()
                .wait_for_fences(&[self.fence], true, timeout)
                .map_err(FenceError::WaitError)
        }
    }

    pub fn reset(&mut self) -> Result<(), FenceError> {
        self.flush_buffers();
        unsafe {
            self.device
                .device()
                .reset_fences(&[self.fence])
                .map_err(FenceError::ResetError)
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_fence(self.fence, None);
        }
    }
}
