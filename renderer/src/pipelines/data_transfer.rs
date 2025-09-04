use std::sync::{Arc, Mutex};

use anyhow::Result as AnyResult;
use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::wrappers::{
    buffer::Buffer,
    command::Command,
    command_buffer::CommandBuffer,
    command_pool::CommandPool,
    fence::Fence,
    image::Image,
    logical_device::{LogicalDevice, QueueType},
};

pub enum DTPInput<'a> {
    CopyToBuffer(&'a [u8], &'a Buffer),
    CopyToImage {
        data: &'a [u8],
        image: &'a Image,
        subresource_layers: vk::ImageSubresourceLayers,
    },
}

pub struct DTP {
    // command_buffers_count: u32,
    command_pool: Arc<CommandPool>,
    // command_buffers: Mutex<Vec<CommandBuffer>>,
    allocator: Arc<Mutex<Allocator>>,
}

impl DTP {
    pub fn new(device: Arc<LogicalDevice>, allocator: Arc<Mutex<Allocator>>) -> AnyResult<Self> {
        let command_pool = CommandPool::new(device, QueueType::Graphics, true).map(Arc::new)?;

        // let command_buffers = CommandBuffer::new(command_pool.clone(), command_buffers_count)
        //     .map(Mutex::new)
        //     .map_err(DTPError::CommandBufferAllocationError)?;

        Ok(Self {
            // command_buffers_count,
            command_pool,
            // command_buffers,
            allocator,
        })
    }

    pub fn create_temp_command_buffer(&self) -> AnyResult<CommandBuffer> {
        Ok(CommandBuffer::new(self.command_pool.clone(), 1)?.remove(0))
    }

    pub fn do_transfers_custom(
        &self,
        transfers: Vec<DTPInput>,
        command_buffer: &CommandBuffer,
    ) -> AnyResult<Buffer> {
        let device = self.command_pool.device();
        let stage_buffer_size: u64 = transfers
            .iter()
            .map(|t| match t {
                DTPInput::CopyToBuffer(data, _) => data.len() as u64,
                DTPInput::CopyToImage { data, .. } => data.len() as u64,
            })
            .sum();

        let mut stage_buffer = Buffer::new(
            device.clone(),
            stage_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        stage_buffer.allocate_memory(self.allocator.clone(), false)?;
        let stage_mem_ptr = stage_buffer.get_allocation_mount_slice()?;
        let mut offset = 0;
        for transfer in &transfers {
            match transfer {
                DTPInput::CopyToBuffer(data, ..) => {
                    let data_len = data.len();
                    stage_mem_ptr[offset..offset + data_len].copy_from_slice(data);
                    offset += data_len;
                }
                DTPInput::CopyToImage { data, .. } => {
                    let data_len = data.len();
                    stage_mem_ptr[offset..offset + data_len].copy_from_slice(data);
                    offset += data_len;
                }
            }
        }

        let mut current_offset = 0;
        let mut commands = vec![];
        for transfer in transfers {
            match transfer {
                DTPInput::CopyToBuffer(data, buffer) => {
                    let data_len = data.len() as u64;
                    if data.len() == 0 {
                        continue;
                    }
                    let copy_region = vk::BufferCopy::default()
                        .src_offset(current_offset)
                        .dst_offset(0)
                        .size(data_len);
                    commands.push(Command::CopyBufferToBuffer {
                        src: &stage_buffer,
                        dst: buffer,
                        regions: vec![copy_region],
                    });
                    current_offset += data_len;
                }
                DTPInput::CopyToImage {
                    data,
                    image,
                    subresource_layers,
                } => {
                    let data_len = data.len() as u64;
                    if data_len == 0 {
                        continue;
                    }
                    let buffer_image_regions = vk::BufferImageCopy::default()
                        .buffer_offset(current_offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(subresource_layers)
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(image.extent());
                    commands.push(Command::CopyBufferToImage {
                        src: &stage_buffer,
                        dst: image,
                        regions: vec![buffer_image_regions],
                    });
                    current_offset += data_len;
                }
            }
        }
        for command in &commands {
            command.record(command_buffer);
        }

        Ok(stage_buffer)
    }

    pub fn do_transfers(&self, transfers: Vec<DTPInput>) -> AnyResult<()> {
        let device = self.command_pool.device();
        let command_buffer = self.create_temp_command_buffer()?;
        command_buffer.begin(true)?;
        let stage_buffer = self.do_transfers_custom(transfers, &command_buffer)?;
        command_buffer.end()?;

        let fence = Fence::new(device.clone(), false)?;

        unsafe {
            device.device().queue_submit(
                device.graphics_queue(),
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer.command_buffer()])],
                fence.fence(),
            )?;
        }
        fence.wait(u64::MAX)?;

        drop(stage_buffer);
        Ok(())
    }
}
