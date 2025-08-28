use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::{LogicalDevice, QueueType};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct CommandPool {
    #[get_copy = "pub"]
    command_pool: vk::CommandPool,
    #[get = "pub"]
    queue_type: QueueType,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl CommandPool {
    pub fn new(device: Arc<LogicalDevice>, queue_type: QueueType) -> Result<Self, vk::Result> {
        let qf_idx = match queue_type {
            QueueType::Graphics => device.graphics_qf_id(),
        };
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_idx)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.device().create_command_pool(&create_info, None)? };

        Ok(Self {
            command_pool,
            queue_type,
            device,
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
