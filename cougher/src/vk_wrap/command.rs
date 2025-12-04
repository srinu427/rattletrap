use std::sync::Arc;

use ash::vk;

use crate::vk_wrap::{
    buffer::Buffer,
    device::Device,
    image_2d::Image2d,
    sync::{Fence, SemStageInfo},
};

#[derive(Debug, thiserror::Error)]
pub enum CommandBufferError {
    #[error("Error creating Vulkan Command Pool: {0}")]
    PoolCreateError(vk::Result),
    #[error("Error allocating Vulkan Command Buffers: {0}")]
    CreateError(vk::Result),
    #[error("Error beginning Vulkan Command Buffer recording: {0}")]
    BeginError(vk::Result),
    #[error("Error ending Vulkan Command Buffer recording: {0}")]
    EndError(vk::Result),
    #[error("Error submitting Vulkan Command Buffer: {0}")]
    SubmitError(vk::Result),
}

pub struct CommandPool {
    pub(crate) cp: vk::CommandPool,
    pub(crate) qf: u32,
    pub(crate) device: Arc<Device>,
}

impl CommandPool {
    pub fn new(device: &Arc<Device>, queue_family: u32) -> Result<Self, CommandBufferError> {
        let cp = unsafe {
            device
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(CommandBufferError::PoolCreateError)?
        };
        Ok(Self {
            cp,
            qf: queue_family,
            device: device.clone(),
        })
    }

    pub fn allocate_cbs(&self, count: u32) -> Result<Vec<CommandBuffer>, CommandBufferError> {
        let cbs = unsafe {
            self.device
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(self.cp)
                        .command_buffer_count(count),
                )
                .map_err(CommandBufferError::CreateError)?
        };
        let cbs = cbs
            .into_iter()
            .map(|cb| CommandBuffer {
                cb,
                device: self.device.clone(),
            })
            .collect();
        Ok(cbs)
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_command_pool(self.cp, None);
        }
    }
}

pub struct CommandBuffer {
    pub(crate) cb: vk::CommandBuffer,
    pub(crate) device: Arc<Device>,
}

pub struct CompositeInput<'a> {
    pub image: &'a Image2d,
    pub in_range: [(f32, f32); 2],
    pub out_range: [(f32, f32); 2],
}

impl CommandBuffer {
    pub fn begin(&self, one_time: bool) -> Result<(), CommandBufferError> {
        let flags = if one_time {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        unsafe {
            self.device
                .device
                .begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default().flags(flags))
                .map_err(CommandBufferError::BeginError)?;
        }
        Ok(())
    }

    pub fn end(&self) -> Result<(), CommandBufferError> {
        unsafe {
            self.device
                .device
                .end_command_buffer(self.cb)
                .map_err(CommandBufferError::EndError)?;
        }
        Ok(())
    }

    pub fn composite_images(&self, dst: &Image2d, inputs: Vec<CompositeInput>) {
        unsafe {
            for inp in inputs {
                let src_offsets = [
                    vk::Offset3D::default()
                        .x((inp.in_range[0].0 * inp.image.extent.width as f32) as _)
                        .y((inp.in_range[0].1 * inp.image.extent.height as f32) as _),
                    vk::Offset3D::default()
                        .x((inp.in_range[1].0 * inp.image.extent.width as f32) as _)
                        .y((inp.in_range[1].1 * inp.image.extent.height as f32) as _)
                        .z(1),
                ];
                let dst_offsets = [
                    vk::Offset3D::default()
                        .x((inp.out_range[0].0 * dst.extent.width as f32) as _)
                        .y((inp.out_range[0].1 * dst.extent.height as f32) as _),
                    vk::Offset3D::default()
                        .x((inp.out_range[1].0 * dst.extent.width as f32) as _)
                        .y((inp.out_range[1].1 * dst.extent.height as f32) as _)
                        .z(1),
                ];
                self.device.device.cmd_blit_image(
                    self.cb,
                    inp.image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit::default()
                        .src_subresource(inp.image.subresource_layers())
                        .src_offsets(src_offsets)
                        .dst_subresource(Image2d::subresource_layers_stc(false, false))
                        .dst_offsets(dst_offsets)],
                    vk::Filter::NEAREST,
                );
            }
        }
    }

    pub fn copy_buf_to_img_2d(&self, buffer: &Buffer, image: &Image2d) {
        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                self.cb,
                buffer.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_extent(
                        vk::Extent3D::default()
                            .width(image.extent.width)
                            .height(image.extent.height)
                            .depth(1),
                    )
                    .image_subresource(Image2d::subresource_layers_stc(false, false))],
            );
        }
    }

    pub fn image_2d_layout_transition(
        &self,
        image: &Image2d,
        old: ImageStageLayout,
        new: ImageStageLayout,
        queue_fam: u32,
    ) {
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                self.cb,
                old.get_stage(),
                new.get_stage(),
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image.image)
                    .subresource_range(image.subresource_range())
                    .old_layout(old.get_layout())
                    .new_layout(new.get_layout())
                    .src_access_mask(old.get_access())
                    .dst_access_mask(new.get_access())
                    .src_queue_family_index(queue_fam)
                    .dst_queue_family_index(queue_fam)],
            );
        }
    }

    pub fn submit(
        &self,
        queue: vk::Queue,
        emit_sems: &[SemStageInfo],
        wait_sems: &[SemStageInfo],
        fence: Option<&Fence>,
    ) -> Result<(), CommandBufferError> {
        let fence_vk = fence.map(|f| f.fence).unwrap_or(vk::Fence::null());
        let emit_sems_vk: Vec<_> = emit_sems.iter().map(|e| e.sem.sem).collect();
        // let emit_stages_vk: Vec<_> = emit_sems.iter().map(|e| e.stage).collect();
        let wait_sems_vk: Vec<_> = wait_sems.iter().map(|e| e.sem.sem).collect();
        let wait_stages_vk: Vec<_> = wait_sems.iter().map(|e| e.stage).collect();
        unsafe {
            self.device
                .device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[self.cb])
                        .signal_semaphores(&emit_sems_vk)
                        .wait_semaphores(&wait_sems_vk)
                        .wait_dst_stage_mask(&wait_stages_vk)],
                    fence_vk,
                )
                .map_err(CommandBufferError::SubmitError)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TransferStageLayout {
    TransferSrc,
    TransferDst,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageStageLayout {
    Undefined,
    Present,
    Transfer(TransferStageLayout),
}

impl ImageStageLayout {
    fn get_stage(&self) -> vk::PipelineStageFlags {
        match self {
            ImageStageLayout::Undefined => vk::PipelineStageFlags::ALL_COMMANDS,
            ImageStageLayout::Present => vk::PipelineStageFlags::ALL_COMMANDS,
            ImageStageLayout::Transfer(_) => vk::PipelineStageFlags::TRANSFER,
        }
    }

    fn get_layout(&self) -> vk::ImageLayout {
        match self {
            ImageStageLayout::Undefined => vk::ImageLayout::UNDEFINED,
            ImageStageLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
            ImageStageLayout::Transfer(transfer_stage_layout) => match transfer_stage_layout {
                TransferStageLayout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                TransferStageLayout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
        }
    }

    fn get_access(&self) -> vk::AccessFlags {
        match self {
            ImageStageLayout::Undefined => vk::AccessFlags::empty(),
            ImageStageLayout::Present => vk::AccessFlags::empty(),
            ImageStageLayout::Transfer(transfer_stage_layout) => match transfer_stage_layout {
                TransferStageLayout::TransferSrc => vk::AccessFlags::TRANSFER_READ,
                TransferStageLayout::TransferDst => vk::AccessFlags::TRANSFER_WRITE,
            },
        }
    }

    pub fn infer_usage(&self) -> vk::ImageUsageFlags {
        match self {
            ImageStageLayout::Undefined => vk::ImageUsageFlags::empty(),
            ImageStageLayout::Present => vk::ImageUsageFlags::empty(),
            ImageStageLayout::Transfer(transfer_stage_layout) => match transfer_stage_layout {
                TransferStageLayout::TransferSrc => vk::ImageUsageFlags::TRANSFER_SRC,
                TransferStageLayout::TransferDst => vk::ImageUsageFlags::TRANSFER_DST,
            },
        }
    }
}
