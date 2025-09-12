use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::wrappers::{command_pool::CommandPool, fence::Fence, semaphore::Semaphore};

#[derive(Debug, Error)]
pub enum CommandBufferError {
    #[error("Command buffer allocation error: {0}")]
    AllocationError(vk::Result),
    #[error("Command buffer begin error: {0}")]
    BeginError(vk::Result),
    #[error("Command buffer end error: {0}")]
    EndError(vk::Result),
    #[error("Command buffer reset error: {0}")]
    ResetError(vk::Result),
    #[error("Command buffer submit error: {0}")]
    SubmitError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct CommandBuffer {
    #[get_copy = "pub"]
    command_buffer: vk::CommandBuffer,
    #[get = "pub"]
    command_pool: Arc<CommandPool>,
}

impl CommandBuffer {
    pub fn new(
        command_pool: Arc<CommandPool>,
        count: u32,
    ) -> Result<Vec<Self>, CommandBufferError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let command_buffers = unsafe {
            command_pool
                .device()
                .device()
                .allocate_command_buffers(&allocate_info)
                .map_err(CommandBufferError::AllocationError)?
        };

        Ok(command_buffers
            .into_iter()
            .map(|cb| CommandBuffer {
                command_buffer: cb,
                command_pool: command_pool.clone(),
            })
            .collect())
    }

    pub fn begin(&self, one_time: bool) -> Result<(), CommandBufferError> {
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
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(CommandBufferError::BeginError)?
        };

        Ok(())
    }

    pub fn end(&self) -> Result<(), CommandBufferError> {
        unsafe {
            self.command_pool
                .device()
                .device()
                .end_command_buffer(self.command_buffer)
                .map_err(CommandBufferError::EndError)?
        };

        Ok(())
    }

    pub fn reset(&self) -> Result<(), CommandBufferError> {
        unsafe {
            self.command_pool
                .device()
                .device()
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(CommandBufferError::ResetError)?
        };

        Ok(())
    }

    pub fn submit(
        &self,
        wait_sems: &[(&Semaphore, vk::PipelineStageFlags2)],
        signal_sems: &[(&Semaphore, vk::PipelineStageFlags2)],
        fence: Option<&Fence>,
    ) -> Result<(), CommandBufferError> {
        let wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfo> = wait_sems
            .iter()
            .map(|(sem, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sem.semaphore())
                    .stage_mask(*stage)
            })
            .collect();

        let signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo> = signal_sems
            .iter()
            .map(|(sem, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sem.semaphore())
                    .stage_mask(*stage)
            })
            .collect();

        let command_buffer_infos = [vk::CommandBufferSubmitInfo::default()
            .command_buffer(self.command_buffer)
            .device_mask(0)];
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);

        unsafe {
            self.command_pool.device().sync2_device().queue_submit2(
                self.command_pool.device().graphics_queue(),
                &[submit_info],
                fence.map(|f| f.fence()).unwrap_or(vk::Fence::null()),
            ).map_err(CommandBufferError::SubmitError)
        }
        
    }
}
