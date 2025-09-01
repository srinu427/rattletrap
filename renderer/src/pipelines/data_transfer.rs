use std::sync::{Arc, Mutex};

use anyhow::Result as AnyResult;
use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::wrappers::{
    buffer::Buffer,
    command_buffer::CommandBuffer,
    command_pool::CommandPool,
    fence::Fence,
    image::Image,
    logical_device::{LogicalDevice, QueueType},
};

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

    pub fn transfer_data(&self, data: &[u8], buffer: &Buffer) -> AnyResult<()> {
        let device = self.command_pool.device();
        let command_buffer = CommandBuffer::new(self.command_pool.clone(), 1)?.remove(0);
        let mut stage_buffer = Buffer::new(
            device.clone(),
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        stage_buffer.allocate_memory(self.allocator.clone(), false)?;
        let stage_mem_ptr = stage_buffer.get_allocation_mount_slice()?;
        stage_mem_ptr.copy_from_slice(data);
        command_buffer.begin(true)?;
        // Record command buffer to copy from stage buffer to destination buffer
        let copy_region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(data.len() as u64);
        unsafe {
            device.device().cmd_copy_buffer(
                command_buffer.command_buffer(),
                stage_buffer.buffer(),
                buffer.buffer(),
                &[copy_region],
            );
        }
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
        Ok(())
    }

    pub fn transfer_data_to_image_2d(&self, data: &[u8], image: &Image) -> AnyResult<()> {
        let device = self.command_pool.device();
        let command_buffer = CommandBuffer::new(self.command_pool.clone(), 1)?.remove(0);
        let mut stage_buffer = Buffer::new(
            device.clone(),
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        stage_buffer.allocate_memory(self.allocator.clone(), false)?;
        let stage_mem_ptr = stage_buffer.get_allocation_mount_slice()?;
        stage_mem_ptr.copy_from_slice(data);
        command_buffer.begin(true)?;
        // Record command buffer to copy from stage buffer to destination image
        let buffer_image_regions = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(image.extent());
        unsafe {
            device.device().cmd_copy_buffer_to_image(
                command_buffer.command_buffer(),
                stage_buffer.buffer(),
                image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[buffer_image_regions],
            );
        }
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
        Ok(())
    }
}
