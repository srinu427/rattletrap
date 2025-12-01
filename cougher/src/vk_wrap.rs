use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationError, MemoryLocation};
use hashbrown::HashMap;
use image::ImageError;

mod buffer;
mod command;
pub mod device;
pub mod image_2d;
pub mod instance;
mod mesh_renderer;
mod pipeline;
mod swapchain;
mod sync;

use crate::render_objs::Mesh;
use crate::vk_wrap::buffer::{Buffer, BufferError};
use crate::vk_wrap::command::{
    CommandBuffer, CommandBufferError, CommandPool, CompositeInput, ImageStageLayout,
    TransferStageLayout,
};
use crate::vk_wrap::device::{Device, DeviceError};
use crate::vk_wrap::image_2d::{Image2d, ImageErrorVk, Sampler};
use crate::vk_wrap::mesh_renderer::{MeshPipeline, MeshPipelineError};
use crate::vk_wrap::swapchain::{Swapchain, SwapchainError};
use crate::vk_wrap::sync::{Fence, Semaphore, SyncError, reset_fences, wait_for_fences};

#[derive(Debug, thiserror::Error)]
pub enum RendererError {
    #[error("Error creating Vulkan Memory Allocator: {0}")]
    AllocatorInitError(AllocationError),
    #[error("Error freeing Vulkan Memory Allocation: {0}")]
    AllocationFreeError(AllocationError),
    #[error("Error creating Vulkan Fence: {0}")]
    FenceError(#[from] SyncError),
    #[error("Error ending Vulkan Command Buffer: {0}")]
    CommandBufferError(#[from] CommandBufferError),
    #[error("Error submitting work to Vulkan Queue: {0}")]
    QueueSubmitError(vk::Result),
    #[error("Error presenting to Swapchain Image: {0}")]
    PresentError(vk::Result),
    #[error("Mesh Pipeline Error: {0}")]
    MeshPipelineError(#[from] MeshPipelineError),
    #[error("Error loading Image from disk: {0}")]
    ImagePathLoadError(#[from] ImageError),
    #[error("Error related to a 2D Image: {0}")]
    Image2dError(#[from] ImageErrorVk),
    #[error("Error related to a Buffer: {0}")]
    BufferError(#[from] BufferError),
    #[error("Error related to Swapchain: {0}")]
    SwapchainError(#[from] SwapchainError),
    #[error("Error related to Device: {0}")]
    DeviceError(#[from] DeviceError),
}

pub enum RendererCommands {
    AddMesh {
        name: String,
        mesh: Mesh,
    },
    RemoveMesh(String),
    RemoveUnusedMeshes,
    AddTexture(String),
    RemoveTexture(String),
    RemoveUnusedTextures,
    DrawAddMeshTexture {
        mesh: String,
        texture: String,
        cast_shadows: bool,
    },
    DrawRemoveMeshTexture {
        mesh: String,
        texture: String,
    },
}

pub struct PerFrameData {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    draw_cmd_buffer: CommandBuffer,
    draw_complete_semaphore: Semaphore,
    draw_complete_fence: Fence,
}

pub struct Renderer {
    textures: HashMap<String, Image2d>,
    sampler: Sampler,
    mesh_pipeline: MeshPipeline,
    swapchain_init_done: bool,
    frame_in_flight: Vec<bool>,
    draw_fences: Vec<Fence>,
    draw_sems: Vec<Semaphore>,
    draw_cmd_buffers: Vec<CommandBuffer>,
    image_acquire_fence: Fence,
    bg_image: Image2d,
    command_pool: CommandPool,
    allocator: Arc<Mutex<Allocator>>,
    swapchain: Swapchain,
    device: Arc<Device>,
}

impl Renderer {
    fn setup_bg_image(
        device: &Arc<Device>,
        allocator: &Arc<Mutex<Allocator>>,
        cmd_buffer: &CommandBuffer,
    ) -> Result<Image2d, RendererError> {
        let bg_image_data = image::open("./default.png")?;
        let bg_image_res = vk::Extent2D::default()
            .width(bg_image_data.width())
            .height(bg_image_data.height());
        let bg_image = Image2d::new(
            &device,
            &allocator,
            MemoryLocation::GpuOnly,
            bg_image_res,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let bg_stage_buffer = Buffer::new_c2g_with_data(
            device,
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            bg_image_data.as_bytes(),
        )?;

        cmd_buffer.begin(true)?;

        cmd_buffer.image_2d_layout_transition(
            &bg_image,
            ImageStageLayout::Undefined,
            ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
            device.g_queue_fam,
        );

        cmd_buffer.copy_buf_to_img_2d(&bg_stage_buffer, &bg_image);

        cmd_buffer.image_2d_layout_transition(
            &bg_image,
            ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
            ImageStageLayout::Transfer(TransferStageLayout::TransferSrc),
            device.g_queue_fam,
        );

        cmd_buffer.end()?;

        let fence = Fence::new(device, false)?;

        unsafe {
            device
                .device
                .queue_submit(
                    device.g_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd_buffer.cb])],
                    fence.fence,
                )
                .map_err(RendererError::QueueSubmitError)?;
        }
        fence.wait(None)?;

        Ok(bg_image)
    }

    pub fn new(device: Device) -> Result<Self, RendererError> {
        let device = Arc::new(device);
        let swapchain = Swapchain::new(&device)?;
        let command_pool = CommandPool::new(&device, device.g_queue_fam)?;
        let allocator = Arc::new(Mutex::new(
            Allocator::new(&AllocatorCreateDesc {
                instance: device.instance.instance.clone(),
                device: device.device.clone(),
                physical_device: device.physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(RendererError::AllocatorInitError)?,
        ));
        let image_acquire_fence = Fence::new(&device, false)?;
        let draw_cmd_buffers = command_pool.allocate_cbs(swapchain.images.len() as _)?;
        let draw_fences: Vec<_> = (0..swapchain.images.len())
            .map(|_| Fence::new(&device, false))
            .collect::<Result<_, _>>()?;
        let draw_sems: Vec<_> = (0..swapchain.images.len())
            .map(|_| Semaphore::new(&device))
            .collect::<Result<_, _>>()?;

        let bg_image = Self::setup_bg_image(&device, &allocator, &draw_cmd_buffers[0])?;

        let mesh_pipeline = MeshPipeline::new(&device)?;
        let sampler = Sampler::new(&device)?;

        Ok(Self {
            textures: HashMap::new(),
            sampler,
            mesh_pipeline,
            swapchain_init_done: false,
            frame_in_flight: vec![false; swapchain.images.len()],
            draw_sems,
            draw_fences,
            draw_cmd_buffers,
            image_acquire_fence,
            bg_image,
            command_pool,
            allocator,
            swapchain,
            device,
        })
    }

    pub fn draw(&mut self) -> Result<(), RendererError> {
        // let start_time = std::time::Instant::now();
        let mut refreshed = false;
        let (image_idx, refreshed) = loop {
            let aquire_out = self.swapchain.acquire_next_img(&self.image_acquire_fence);

            let (idx, is_suboptimal) = match aquire_out {
                Ok((i, s)) => (Some(i), s),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (None, true),
                Err(e) => {
                    return Err(RendererError::SwapchainError(
                        SwapchainError::AcquireNextImageError(e),
                    ));
                }
            };

            if is_suboptimal {
                let in_flight_fences: Vec<_> = self
                    .draw_fences
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| self.frame_in_flight[*i])
                    .map(|(_, f)| f)
                    .collect();
                wait_for_fences(&self.device.device, &in_flight_fences, None)?;
                reset_fences(&self.device.device, &in_flight_fences)?;
                self.frame_in_flight = vec![false; self.swapchain.images.len()];
                self.swapchain.refresh_swapchain_res()?;
                refreshed = true;
                if idx.is_some() {
                    self.image_acquire_fence.wait(None)?;
                    self.image_acquire_fence.reset()?;
                }
                continue;
            }
            if let Some(img_idx) = idx {
                self.image_acquire_fence.wait(None)?;
                self.image_acquire_fence.reset()?;
                break (img_idx, refreshed);
            }
        };
        // let aquire_time = start_time.elapsed().as_millis();

        self.swapchain_init_done &= !refreshed;
        let idx = image_idx as usize;
        let cmd_buffer = &self.draw_cmd_buffers[idx];
        let swapchain_image = &self.swapchain.images[idx];

        if self.frame_in_flight[idx] {
            self.draw_fences[idx].wait(None)?;
            self.draw_fences[idx].reset()?;
            self.frame_in_flight[idx] = false;
        }

        cmd_buffer.begin(false)?;

        if !self.swapchain_init_done {
            for (i, swi) in self.swapchain.images.iter().enumerate() {
                if i != idx {
                    cmd_buffer.image_2d_layout_transition(
                        swi,
                        ImageStageLayout::Undefined,
                        ImageStageLayout::Present,
                        self.device.g_queue_fam,
                    );
                } else {
                    cmd_buffer.image_2d_layout_transition(
                        swi,
                        ImageStageLayout::Undefined,
                        ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
                        self.device.g_queue_fam,
                    );
                };
            }
        } else {
            cmd_buffer.image_2d_layout_transition(
                swapchain_image,
                ImageStageLayout::Present,
                ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
                self.device.g_queue_fam,
            );
        }

        cmd_buffer.composite_images(
            swapchain_image,
            vec![CompositeInput {
                image: &self.bg_image,
                in_range: [(0.0, 0.0), (1.0, 1.0)],
                out_range: [(0.0, 0.0), (1.0, 1.0)],
            }],
        );

        cmd_buffer.image_2d_layout_transition(
            swapchain_image,
            ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
            ImageStageLayout::Present,
            self.device.g_queue_fam,
        );

        cmd_buffer.end()?;

        cmd_buffer.submit(
            self.device.g_queue,
            &[self.draw_sems[idx].stage_info(vk::PipelineStageFlags::ALL_COMMANDS)],
            &[],
            Some(&self.draw_fences[idx]),
        )?;
        self.frame_in_flight[idx] = true;

        self.swapchain_init_done = true;
        self.swapchain.present_image(
            image_idx,
            &[self.draw_sems[idx].stage_info(vk::PipelineStageFlags::ALL_COMMANDS)],
        )?;
        // print!(
        //     "draw time: {} ms. acquire time: {} ms\r",
        //     start_time.elapsed().as_millis(),
        //     aquire_time
        // );
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
        }
    }
}
