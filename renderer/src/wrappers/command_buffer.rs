use std::sync::Arc;

use ash::vk;

use crate::wrappers::{command::RenderPassCommand, command_pool::CommandPool};

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

    pub fn begin(&self, one_time: bool) -> Result<(), vk::Result> {
        let flags = if one_time {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);

        unsafe {
            self.command_pool
                .device()
                .device()
                .begin_command_buffer(self.command_buffer, &begin_info)?
        };

        Ok(())
    }

    pub fn end(&self) -> Result<(), vk::Result> {
        unsafe {
            self.command_pool
                .device()
                .device()
                .end_command_buffer(self.command_buffer)?
        };

        Ok(())
    }

    pub fn reset(&self) -> Result<(), vk::Result> {
        unsafe {
            self.command_pool
                .device()
                .device()
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?
        };

        Ok(())
    }

    fn apply_render_pass_commands(
        &self,
        pipelines: &[vk::Pipeline],
        pipeline_layouts: &[vk::PipelineLayout],
        commands: &[RenderPassCommand],
    ) {
        for command in commands {
            command.apply_command(
                self,
                pipelines,
                pipeline_layouts,
            );
        }
    }
}
