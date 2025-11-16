use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationError, MemoryLocation};
use image::ImageError;

mod buffer;
mod command;
mod device;
mod image_vk;
mod instance;
mod sync;

use crate::vk12::buffer::{Buffer, BufferError};
use crate::vk12::command::{allocate_command_buffers, create_command_pool};
use crate::vk12::device::Vk12Device;
use crate::vk12::image_vk::{ImageErrorVk, image_subresource_layers_2d, new_image_2d};
use crate::vk12::sync::create_fence;

pub struct CompositeInput<'a> {
    pub image: &'a vk::Image,
    pub image_res: vk::Extent2D,
    pub in_range: [(f32, f32); 2],
    pub out_range: [(f32, f32); 2],
}

fn composite_images(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    dst: vk::Image,
    dst_res: vk::Extent2D,
    inputs: Vec<CompositeInput>,
) {
    unsafe {
        for inp in inputs {
            let src_offsets = [
                vk::Offset3D::default()
                    .x((inp.in_range[0].0 * inp.image_res.width as f32) as _)
                    .y((inp.in_range[0].1 * inp.image_res.height as f32) as _),
                vk::Offset3D::default()
                    .x((inp.in_range[1].0 * inp.image_res.width as f32) as _)
                    .y((inp.in_range[1].1 * inp.image_res.height as f32) as _)
                    .z(1),
            ];
            let dst_offsets = [
                vk::Offset3D::default()
                    .x((inp.out_range[0].0 * dst_res.width as f32) as _)
                    .y((inp.out_range[0].1 * dst_res.height as f32) as _),
                vk::Offset3D::default()
                    .x((inp.out_range[1].0 * dst_res.width as f32) as _)
                    .y((inp.out_range[1].1 * dst_res.height as f32) as _)
                    .z(1),
            ];
            device.cmd_blit_image(
                cmd_buffer,
                *inp.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(image_subresource_layers_2d(false, false))
                    .src_offsets(src_offsets)
                    .dst_subresource(image_subresource_layers_2d(false, false))
                    .dst_offsets(dst_offsets)],
                vk::Filter::NEAREST,
            );
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Vk12RendererError {
    #[error("Error creating Vulkan Memory Allocator: {0}")]
    AllocatorInitError(AllocationError),
    #[error("Error freeing Vulkan Memory Allocation: {0}")]
    AllocationFreeError(AllocationError),
    #[error("Error creating Vulkan Command Pool: {0}")]
    CommandPoolCreateError(vk::Result),
    #[error("Error creating Vulkan Fence: {0}")]
    FenceCreateError(vk::Result),
    #[error("Error waiting for Vulkan Fences: {0}")]
    FenceWaitError(vk::Result),
    #[error("Error allocating Vulkan Command Buffers: {0}")]
    CommandBufferAllocateError(vk::Result),
    #[error("Error beginning Vulkan Command Buffer: {0}")]
    CommandBufferBeginError(vk::Result),
    #[error("Error ending Vulkan Command Buffer: {0}")]
    CommandBufferEndError(vk::Result),
    #[error("Error submitting work to Vulkan Queue: {0}")]
    QueueSubmitError(vk::Result),
    #[error("Error loading Image from disk: {0}")]
    ImagePathLoadError(#[from] ImageError),
    #[error("Error related to a 2D Image: {0}")]
    Image2dError(#[from] ImageErrorVk),
    #[error("Error related to a Buffer: {0}")]
    BufferError(#[from] BufferError),
}

pub struct Vk12Renderer {
    swapchain_init_done: bool,
    draw_fences: Vec<vk::Fence>,
    draw_cmd_buffers: Vec<vk::CommandBuffer>,
    image_acquire_fence: vk::Fence,
    bg_image_mem: Allocation,
    bg_image: vk::Image,
    command_pool: vk::CommandPool,
    allocator: Allocator,
    device: Vk12Device,
}

impl Vk12Renderer {
    fn setup_bg_image(
        device: &Vk12Device,
        allocator: &mut Allocator,
        cmd_buffer: vk::CommandBuffer,
    ) -> Result<(vk::Image, Allocation), Vk12RendererError> {
        let bg_image_data = image::open("./default.png")?;
        let bg_image_res = vk::Extent2D::default()
            .width(bg_image_data.width())
            .height(bg_image_data.height());
        let (bg_image, bg_image_mem) = new_image_2d(
            &device.device,
            allocator,
            MemoryLocation::GpuOnly,
            bg_image_res,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let mut bg_stage_buffer = match Buffer::new_cpu_to_gpu_with_data(
            &device.device,
            allocator,
            bg_image_data.as_bytes(),
        ) {
            Ok(b) => b,
            Err(e) => {
                allocator
                    .free(bg_image_mem)
                    .map_err(Vk12RendererError::AllocationFreeError)?;
                return Err(e.into());
            }
        };

        unsafe {
            if let Err(e) = device
                .device
                .begin_command_buffer(
                    cmd_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(Vk12RendererError::CommandBufferBeginError)
            {
                allocator
                    .free(bg_image_mem)
                    .map_err(Vk12RendererError::AllocationFreeError)?;
                bg_stage_buffer.cleanup(&device.device, allocator);
                return Err(e);
            }

            device.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(bg_image.read())
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(device.g_queue_fam)
                    .dst_queue_family_index(device.g_queue_fam)],
            );

            device.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                bg_stage_buffer.buffer,
                bg_image.read(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_extent(
                        vk::Extent3D::default()
                            .width(bg_image_res.width)
                            .height(bg_image_res.height)
                            .depth(1),
                    )
                    .image_subresource(image_subresource_layers_2d(false, false))],
            );

            device.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(bg_image.read())
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(device.g_queue_fam)
                    .dst_queue_family_index(device.g_queue_fam)],
            );

            if let Err(e) = device
                .device
                .end_command_buffer(cmd_buffer)
                .map_err(Vk12RendererError::CommandBufferEndError)
            {
                allocator.free(bg_image_mem);
                bg_stage_buffer.cleanup(&device.device, allocator);
                return Err(e);
            }

            let fence =
                match create_fence(&device.device).map_err(Vk12RendererError::FenceCreateError) {
                    Ok(f) => f,
                    Err(e) => {
                        allocator.free(bg_image_mem);
                        bg_stage_buffer.cleanup(&device.device, allocator);
                        return Err(e);
                    }
                };
            if let Err(e) = device
                .device
                .queue_submit(
                    device.g_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                    fence.read(),
                )
                .map_err(Vk12RendererError::QueueSubmitError)
            {
                allocator.free(bg_image_mem);
                bg_stage_buffer.cleanup(&device.device, allocator);
                return Err(e);
            };
            if let Err(e) = device
                .device
                .wait_for_fences(&[fence.read()], true, u64::MAX)
                .map_err(Vk12RendererError::FenceWaitError)
            {
                allocator.free(bg_image_mem);
                bg_stage_buffer.cleanup(&device.device, allocator);
                return Err(e);
            };
        }

        bg_stage_buffer.cleanup(&device.device, allocator);
        Ok((bg_image.take(), bg_image_mem))
    }

    pub fn new(device: Vk12Device) -> Result<Self, Vk12RendererError> {
        let command_pool = create_command_pool(&device.device, device.g_queue_fam)
            .map_err(Vk12RendererError::CommandPoolCreateError)?;
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: device.instance.instance.clone(),
            device: device.device.clone(),
            physical_device: device.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(Vk12RendererError::AllocatorInitError)?;
        let image_acquire_fence =
            create_fence(&device.device).map_err(Vk12RendererError::FenceCreateError)?;
        let draw_cmd_buffers = allocate_command_buffers(
            &device.device,
            command_pool.read(),
            device.swapchain_data.images.len() as _,
        )
        .map_err(Vk12RendererError::CommandBufferAllocateError)?;
        let draw_fences: Vec<_> = (0..device.swapchain_data.images.len())
            .map(|_| create_fence(&device.device))
            .collect::<Result<_, _>>()
            .map_err(Vk12RendererError::FenceCreateError)?;

        let (bg_image, bg_image_mem) =
            Self::setup_bg_image(&device, &mut allocator, draw_cmd_buffers[0])?;

        Ok(Self {
            swapchain_init_done: false,
            draw_fences: draw_fences.into_iter().map(|f| f.take()).collect(),
            draw_cmd_buffers,
            image_acquire_fence: image_acquire_fence.take(),
            bg_image_mem,
            bg_image,
            command_pool: command_pool.take(),
            allocator,
            device,
        })
    }

    pub fn draw(&mut self) -> Result<(), Vk12RendererError> {
        todo!()
    }
}

impl Drop for Vk12Renderer {
    fn drop(&mut self) {
        unsafe {
            for f in self.draw_fences.drain(..) {
                self.device.device.destroy_fence(f, None);
            }
            self.device
                .device
                .destroy_fence(self.image_acquire_fence, None);
            self.device
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
