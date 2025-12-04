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

use crate::render_objs::{Mesh, MeshTexture};
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
    need_mesh_data_rebuild: bool,
    frame_in_flight: bool,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    draw_cmd_buffer: CommandBuffer,
    draw_complete_semaphore: Semaphore,
    draw_complete_fence: Fence,
}

impl PerFrameData {
    pub fn new(
        device: &Arc<Device>,
        allocator: &Arc<Mutex<Allocator>>,
        cmd_pool: &CommandPool,
    ) -> Result<Self, RendererError> {
        let vertex_buffer = Buffer::new(
            device,
            allocator,
            MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            32 * 1024 * 1024,
        )?;
        let index_buffer = Buffer::new(
            device,
            allocator,
            MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            32 * 1024 * 1024,
        )?;
        let draw_cmd_buffer = cmd_pool.allocate_cbs(1)?.remove(0);
        let draw_complete_semaphore = Semaphore::new(device)?;
        let draw_complete_fence = Fence::new(device, false)?;
        Ok(Self {
            need_mesh_data_rebuild: true,
            frame_in_flight: false,
            vertex_buffer,
            index_buffer,
            draw_cmd_buffer,
            draw_complete_semaphore,
            draw_complete_fence,
        })
    }
}

pub struct Renderer {
    mesh_draw_list: HashMap<String, (String, String)>,
    mesh_textures: HashMap<String, MeshTexture>,
    meshes: HashMap<String, Mesh>,
    sampler: Sampler,
    mesh_pipeline: MeshPipeline,
    swapchain_init_done: bool,
    per_frame_datas: Vec<PerFrameData>,
    image_acquire_fence: Fence,
    bg_image: Image2d,
    command_pool: CommandPool,
    allocator: Arc<Mutex<Allocator>>,
    swapchain: Swapchain,
    device: Arc<Device>,
}

impl Renderer {
    fn setup_bg_image(
        allocator: &Arc<Mutex<Allocator>>,
        cmd_buffer: &CommandBuffer,
    ) -> Result<Image2d, RendererError> {
        let fence = Fence::new(&cmd_buffer.device, false)?;
        let bg_image = Self::load_image_to_gpu(
            allocator,
            cmd_buffer,
            &fence,
            "./default.png",
            ImageStageLayout::Transfer(TransferStageLayout::TransferSrc),
        )?;

        Ok(bg_image)
    }

    fn load_image_to_gpu(
        allocator: &Arc<Mutex<Allocator>>,
        cmd_buffer: &CommandBuffer,
        fence: &Fence,
        path: &str,
        final_layout: ImageStageLayout,
    ) -> Result<Image2d, RendererError> {
        let image_data = image::open(path)?;
        let image_res = vk::Extent2D::default()
            .width(image_data.width())
            .height(image_data.height());
        let device = &cmd_buffer.device;
        let usage = vk::ImageUsageFlags::TRANSFER_DST | final_layout.infer_usage();
        let image = Image2d::new(
            &device,
            &allocator,
            MemoryLocation::GpuOnly,
            image_res,
            vk::Format::R8G8B8A8_UNORM,
            usage,
        )?;
        let stage_buffer = Buffer::new_c2g_with_data(
            device,
            allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            image_data.as_bytes(),
        )?;

        cmd_buffer.begin(true)?;

        cmd_buffer.image_2d_layout_transition(
            &image,
            ImageStageLayout::Undefined,
            ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
            device.g_queue_fam,
        );

        cmd_buffer.copy_buf_to_img_2d(&stage_buffer, &image);

        cmd_buffer.image_2d_layout_transition(
            &image,
            ImageStageLayout::Transfer(TransferStageLayout::TransferDst),
            final_layout,
            device.g_queue_fam,
        );

        cmd_buffer.end()?;

        cmd_buffer.submit(device.g_queue, &[], &[], Some(fence))?;

        fence.wait(None)?;
        fence.reset()?;

        Ok(image)
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

        let per_frame_datas: Vec<_> = (0..swapchain.images.len())
            .map(|_| PerFrameData::new(&device, &allocator, &command_pool))
            .collect::<Result<_, _>>()?;

        let bg_image = Self::setup_bg_image(&allocator, &per_frame_datas[0].draw_cmd_buffer)?;

        let mesh_pipeline = MeshPipeline::new(&device)?;
        let sampler = Sampler::new(&device)?;

        Ok(Self {
            mesh_draw_list: HashMap::new(),
            mesh_textures: HashMap::new(),
            meshes: HashMap::new(),
            sampler,
            mesh_pipeline,
            swapchain_init_done: false,
            per_frame_datas,
            image_acquire_fence,
            bg_image,
            command_pool,
            allocator,
            swapchain,
            device,
        })
    }

    pub fn resize(&mut self) -> Result<(), RendererError> {
        let in_flight_fences: Vec<_> = self
            .per_frame_datas
            .iter()
            .filter(|p| p.frame_in_flight)
            .map(|p| &p.draw_complete_fence)
            .collect();
        wait_for_fences(&in_flight_fences, None)?;
        reset_fences(&in_flight_fences)?;
        for p in self.per_frame_datas.iter_mut() {
            p.frame_in_flight = false;
        }
        self.swapchain.refresh_swapchain_res()?;
        self.swapchain_init_done = false;
        Ok(())
    }

    pub fn draw(&mut self) -> Result<u128, RendererError> {
        let start_time = std::time::Instant::now();
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
                    .per_frame_datas
                    .iter()
                    .filter(|p| p.frame_in_flight)
                    .map(|p| &p.draw_complete_fence)
                    .collect();
                wait_for_fences(&in_flight_fences, None)?;
                reset_fences(&in_flight_fences)?;
                for p in self.per_frame_datas.iter_mut() {
                    p.frame_in_flight = false;
                }
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
        let swapchain_image = &self.swapchain.images[idx];

        if self.per_frame_datas[idx].frame_in_flight {
            self.per_frame_datas[idx].draw_complete_fence.wait(None)?;
            self.per_frame_datas[idx].draw_complete_fence.reset()?;
            self.per_frame_datas[idx].frame_in_flight = false;
        }

        let cmd_buffer = &self.per_frame_datas[idx].draw_cmd_buffer;
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
            &[self.per_frame_datas[idx]
                .draw_complete_semaphore
                .stage_info(vk::PipelineStageFlags::ALL_COMMANDS)],
            &[],
            Some(&self.per_frame_datas[idx].draw_complete_fence),
        )?;
        self.per_frame_datas[idx].frame_in_flight = true;

        self.swapchain_init_done = true;
        self.swapchain.present_image(
            image_idx,
            &[self.per_frame_datas[idx]
                .draw_complete_semaphore
                .stage_info(vk::PipelineStageFlags::ALL_COMMANDS)],
        )?;
        // print!(
        //     "draw time: {} ms. acquire time: {} ms\r",
        //     start_time.elapsed().as_millis(),
        //     aquire_time
        // );
        Ok(start_time.elapsed().as_millis())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
        }
    }
}
