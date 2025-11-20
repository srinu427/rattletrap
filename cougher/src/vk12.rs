use std::mem::ManuallyDrop;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationError, MemoryLocation};
use image::ImageError;

mod buffer;
mod command;
pub mod device;
mod image_vk;
pub mod instance;
mod sync;

use crate::vk12::buffer::{BufferError, new_c2g_buffer_with_data};
use crate::vk12::command::{
    CompositeInput, allocate_command_buffers, begin_cmd_buffer, composite_images,
    create_command_pool, end_cmd_buffer,
};
use crate::vk12::device::{Vk12Device, Vk12DeviceError};
use crate::vk12::image_vk::{
    ImageErrorVk, image_subresource_layers_2d, image_subresource_range_2d, new_image_2d,
};
use crate::vk12::sync::{create_fence, reset_fences, wait_for_fences};

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
    #[error("Error resetting for Vulkan Fences: {0}")]
    FenceResetError(vk::Result),
    #[error("Error allocating Vulkan Command Buffers: {0}")]
    CommandBufferAllocateError(vk::Result),
    #[error("Error beginning Vulkan Command Buffer: {0}")]
    CommandBufferBeginError(vk::Result),
    #[error("Error ending Vulkan Command Buffer: {0}")]
    CommandBufferEndError(vk::Result),
    #[error("Error submitting work to Vulkan Queue: {0}")]
    QueueSubmitError(vk::Result),
    #[error("Error presenting to Swapchain Image: {0}")]
    PresentError(vk::Result),
    #[error("Error loading Image from disk: {0}")]
    ImagePathLoadError(#[from] ImageError),
    #[error("Error related to a 2D Image: {0}")]
    Image2dError(#[from] ImageErrorVk),
    #[error("Error related to a Buffer: {0}")]
    BufferError(#[from] BufferError),
    #[error("Error related to Device: {0}")]
    DeviceError(#[from] Vk12DeviceError),
}

pub struct Vk12Renderer {
    swapchain_init_done: bool,
    draw_fences: Vec<vk::Fence>,
    draw_cmd_buffers: Vec<vk::CommandBuffer>,
    image_acquire_fence: vk::Fence,
    bg_image_mem: ManuallyDrop<Allocation>,
    bg_image_res: vk::Extent2D,
    bg_image: vk::Image,
    command_pool: vk::CommandPool,
    allocator: Allocator,
    device: Vk12Device,
}

impl Vk12Renderer {
    fn free_allocs(allocator: &mut Allocator, allocations: Vec<Allocation>) {
        for alloc in allocations {
            let _ = allocator
                .free(alloc)
                .inspect_err(|e| eprintln!("error freeing gpu memory: {e}"));
        }
    }

    fn setup_bg_image(
        device: &Vk12Device,
        allocator: &mut Allocator,
        cmd_buffer: vk::CommandBuffer,
    ) -> Result<(vk::Extent2D, vk::Image, Allocation), Vk12RendererError> {
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
        let (bg_stage_buffer, bg_stage_buffer_mem) = match new_c2g_buffer_with_data(
            &device.device,
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            bg_image_data.as_bytes(),
        ) {
            Ok(b) => b,
            Err(e) => {
                Self::free_allocs(allocator, vec![bg_image_mem]);
                return Err(e.into());
            }
        };

        unsafe {
            if let Err(e) = device.device.begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ) {
                Self::free_allocs(allocator, vec![bg_image_mem, bg_stage_buffer_mem]);
                return Err(Vk12RendererError::CommandBufferBeginError(e));
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
                    .subresource_range(image_subresource_range_2d(false, false))
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(device.g_queue_fam)
                    .dst_queue_family_index(device.g_queue_fam)],
            );

            device.device.cmd_copy_buffer_to_image(
                cmd_buffer,
                bg_stage_buffer.read(),
                bg_image.read(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(bg_image.read())
                    .subresource_range(image_subresource_range_2d(false, false))
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(device.g_queue_fam)
                    .dst_queue_family_index(device.g_queue_fam)],
            );

            if let Err(e) = device.device.end_command_buffer(cmd_buffer) {
                Self::free_allocs(allocator, vec![bg_image_mem, bg_stage_buffer_mem]);
                return Err(Vk12RendererError::CommandBufferEndError(e));
            }

            let fence = match create_fence(&device.device) {
                Ok(f) => f,
                Err(e) => {
                    Self::free_allocs(allocator, vec![bg_image_mem, bg_stage_buffer_mem]);
                    return Err(Vk12RendererError::FenceCreateError(e));
                }
            };
            if let Err(e) = device.device.queue_submit(
                device.g_queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                fence.read(),
            ) {
                Self::free_allocs(allocator, vec![bg_image_mem, bg_stage_buffer_mem]);
                return Err(Vk12RendererError::QueueSubmitError(e));
            };
            if let Err(e) = device
                .device
                .wait_for_fences(&[fence.read()], true, u64::MAX)
            {
                Self::free_allocs(allocator, vec![bg_image_mem, bg_stage_buffer_mem]);
                return Err(Vk12RendererError::FenceWaitError(e));
            };
        }

        Self::free_allocs(allocator, vec![bg_stage_buffer_mem]);
        Ok((bg_image_res, bg_image.take(), bg_image_mem))
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

        let (bg_image_res, bg_image, bg_image_mem) =
            Self::setup_bg_image(&device, &mut allocator, draw_cmd_buffers[0])?;

        Ok(Self {
            swapchain_init_done: false,
            draw_fences: draw_fences.into_iter().map(|f| f.take()).collect(),
            draw_cmd_buffers,
            image_acquire_fence: image_acquire_fence.take(),
            bg_image_res,
            bg_image_mem: ManuallyDrop::new(bg_image_mem),
            bg_image,
            command_pool: command_pool.take(),
            allocator,
            device,
        })
    }

    pub fn draw(&mut self) -> Result<(), Vk12RendererError> {
        let (image_idx, refreshed) = self.device.acquire_next_ws_img(self.image_acquire_fence)?;
        // wait_for_fences(&self.device.device, &[self.image_acquire_fence], None)
        //     .map_err(Vk12RendererError::FenceWaitError)?;
        // reset_fences(&self.device.device, &[self.image_acquire_fence])
        //     .map_err(Vk12RendererError::FenceResetError)?;

        self.swapchain_init_done &= !refreshed;
        let idx = image_idx as usize;
        let cmd_buffer = self.draw_cmd_buffers[idx];
        begin_cmd_buffer(&self.device.device, cmd_buffer, false)
            .map_err(Vk12RendererError::CommandBufferBeginError)?;

        if !self.swapchain_init_done {
            for (i, &swi) in self.device.swapchain_data.images.iter().enumerate() {
                if i != idx {
                    unsafe {
                        self.device.device.cmd_pipeline_barrier(
                            cmd_buffer,
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::DependencyFlags::BY_REGION,
                            &[],
                            &[],
                            &[vk::ImageMemoryBarrier::default()
                                .image(swi)
                                .subresource_range(image_subresource_range_2d(false, false))
                                .old_layout(vk::ImageLayout::UNDEFINED)
                                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                                .src_access_mask(vk::AccessFlags::empty())
                                .dst_access_mask(vk::AccessFlags::empty())
                                .src_queue_family_index(self.device.g_queue_fam)
                                .dst_queue_family_index(self.device.g_queue_fam)],
                        );
                    }
                } else {
                    unsafe {
                        self.device.device.cmd_pipeline_barrier(
                            cmd_buffer,
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::BY_REGION,
                            &[],
                            &[],
                            &[vk::ImageMemoryBarrier::default()
                                .image(swi)
                                .subresource_range(image_subresource_range_2d(false, false))
                                .old_layout(vk::ImageLayout::UNDEFINED)
                                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                .src_access_mask(vk::AccessFlags::empty())
                                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .src_queue_family_index(self.device.g_queue_fam)
                                .dst_queue_family_index(self.device.g_queue_fam)],
                        );
                    }
                };
            }
        } else {
            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(self.device.swapchain_data.images[idx])
                        .subresource_range(image_subresource_range_2d(false, false))
                        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(self.device.g_queue_fam)
                        .dst_queue_family_index(self.device.g_queue_fam)],
                );
            }
        }

        composite_images(
            &self.device.device,
            cmd_buffer,
            self.device.swapchain_data.images[idx],
            self.device.swapchain_data.extent,
            vec![CompositeInput {
                image: self.bg_image,
                image_res: self.bg_image_res,
                in_range: [(0.0, 0.0), (1.0, 1.0)],
                out_range: [(0.0, 0.0), (1.0, 1.0)],
            }],
        );
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cmd_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(self.device.swapchain_data.images[idx])
                    .subresource_range(image_subresource_range_2d(false, false))
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(self.device.g_queue_fam)
                    .dst_queue_family_index(self.device.g_queue_fam)],
            );
        }
        end_cmd_buffer(&self.device.device, cmd_buffer)
            .map_err(Vk12RendererError::CommandBufferEndError)?;

        unsafe {
            self.device
                .device
                .queue_submit(
                    self.device.g_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer])],
                    self.draw_fences[idx],
                )
                .map_err(Vk12RendererError::QueueSubmitError)?;
        }
        wait_for_fences(&self.device.device, &[self.draw_fences[idx]], None)
            .map_err(Vk12RendererError::FenceWaitError)?;
        reset_fences(&self.device.device, &[self.draw_fences[idx]])
            .map_err(Vk12RendererError::FenceResetError)?;
        self.swapchain_init_done = true;
        unsafe {
            self.device
                .swapchain_device
                .queue_present(
                    self.device.g_queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.device.swapchain_data.swapchain])
                        .image_indices(&[image_idx]),
                )
                .map_err(Vk12RendererError::PresentError)?;
        }
        Ok(())
    }
}

impl Drop for Vk12Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
            let altn = ManuallyDrop::take(&mut self.bg_image_mem);
            let _ = self.allocator.free(altn);
            self.device.device.destroy_image(self.bg_image, None);
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
