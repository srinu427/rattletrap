use std::sync::{Arc, Mutex, PoisonError};

use anyhow::Context;
use ash::vk;

use crate::vkraii::{device::DeviceDropper, resource::BufferRaii};

pub struct CommandPoolDropper {
    pub command_buffers: Mutex<Vec<vk::CommandBuffer>>,
    pub command_pool: vk::CommandPool,
    device_d: Arc<DeviceDropper>,
}

impl Drop for CommandPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.device_d
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct CommandPoolRaii {
    command_pool_d: Arc<CommandPoolDropper>,
}

impl CommandPoolRaii {
    pub fn new(device_d: &Arc<DeviceDropper>) -> anyhow::Result<Self> {
        let command_pool = unsafe {
            device_d
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(device_d.graphics_qf)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .with_context(|| "command pool creation failed")?
        };
        Ok(Self {
            command_pool_d: Arc::new(CommandPoolDropper {
                command_buffers: Default::default(),
                command_pool,
                device_d: device_d.clone(),
            }),
        })
    }

    pub fn get_cb(&self) -> anyhow::Result<CommandBufferRaii> {
        let cached_cb = self
            .command_pool_d
            .command_buffers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let cb = match cached_cb {
            Some(t) => t,
            None => unsafe {
                self.command_pool_d
                    .device_d
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(1)
                            .command_pool(self.command_pool_d.command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )?[0]
            },
        };
        let mut cbr = CommandBufferRaii {
            preserve_buffers: Default::default(),
            command_buffer: cb,
            is_recording: false,
            command_pool_d: self.command_pool_d.clone(),
        };
        cbr.begin()?;
        Ok(cbr)
    }
}

pub struct CommandBufferRaii {
    pub preserve_buffers: Vec<BufferRaii>,
    pub command_buffer: vk::CommandBuffer,
    pub is_recording: bool,
    command_pool_d: Arc<CommandPoolDropper>,
}

impl CommandBufferRaii {
    pub fn begin(&mut self) -> anyhow::Result<()> {
        if !self.is_recording {
            unsafe {
                self.command_pool_d
                    .device_d
                    .device
                    .begin_command_buffer(
                        self.command_buffer,
                        &vk::CommandBufferBeginInfo::default(),
                    )
                    .with_context(|| {
                        format!("beginning command buffer: {:?} failed", self.command_buffer)
                    })?;
            }
            self.is_recording = true;
        }
        Ok(())
    }

    pub fn end(&mut self) -> anyhow::Result<()> {
        if self.is_recording {
            unsafe {
                self.command_pool_d
                    .device_d
                    .device
                    .end_command_buffer(self.command_buffer)
                    .with_context(|| {
                        format!("ending command buffer: {:?} failed", self.command_buffer)
                    })?;
            }
            self.is_recording = false;
        }
        Ok(())
    }
}

impl Drop for CommandBufferRaii {
    fn drop(&mut self) {
        self.end().ok();
        self.command_pool_d
            .command_buffers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.command_buffer);
    }
}

pub struct Task {
    pub command_buffers: Vec<CommandBufferRaii>,
    pub task_val: u64,
}
