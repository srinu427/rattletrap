use std::sync::Arc;

use ash::vk;

use crate::wrappers::command_pool::CommandPool;

#[derive(getset::Getters, getset::CopyGetters)]
pub struct CommandBuffer {
    #[get_copy = "pub"]
    command_buffer: vk::CommandBuffer,
    #[get = "pub"]
    command_pool: Arc<CommandPool>,
}

impl CommandBuffer {
    pub fn new(command_pool: Arc<CommandPool>, count: u32) -> Result<Vec<Self>, vk::Result> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let command_buffers = unsafe {
            command_pool
                .device()
                .device()
                .allocate_command_buffers(&allocate_info)?
        };

        Ok(command_buffers
            .into_iter()
            .map(|cb| CommandBuffer {
                command_buffer: cb,
                command_pool: command_pool.clone(),
            })
            .collect())
    }
}

// impl Drop for CommandBuffer {
//     fn drop(&mut self) {
//         unsafe {
//             self.command_pool.device().device().free_command_buffers(
//                 self.command_pool.command_pool(),
//                 &[self.command_buffer],
//             );
//         }
//     }
// }
