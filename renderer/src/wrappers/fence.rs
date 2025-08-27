use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Fence {
    #[get_copy = "pub"]
    fence: vk::Fence,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Fence {
    pub fn new(device: Arc<LogicalDevice>, signaled: bool) -> Result<Self, vk::Result> {
        let create_info = vk::FenceCreateInfo::default().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });

        let fence = unsafe { device.device().create_fence(&create_info, None)? };

        Ok(Self { fence, device })
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_fence(self.fence, None);
        }
    }
}
