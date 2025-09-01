pub mod pipelines;
pub mod renderables;
pub mod wrappers;

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use hashbrown::HashMap;
use thiserror::Error;
use winit::window::Window;

use crate::{
    pipelines::textured_tri_mesh::{TTMP, TTMPAttachments, TTMPError, TTMPSets},
    renderables::{texture::Texture, tri_mesh::TriMesh},
    wrappers::{
        command_buffer::{CommandBuffer, CommandBufferError},
        command_pool::{CommandPool, CommandPoolError},
        descriptor_pool::{DescriptorPool, DescriptorPoolError},
        instance::{Instance, InstanceError},
        logical_device::{LogicalDevice, LogicalDeviceError, QueueType},
        swapchain::{Swapchain, SwapchainError},
    },
};

pub struct TTPMRenderable {
    mesh: String,
    texture: String,
}

#[derive(Debug, Error)]
pub enum RendererInitError {
    #[error("Instance error: {0}")]
    InstanceError(#[from] InstanceError),
    #[error("Logical device creation error: {0}")]
    LogicalDeviceInitError(#[from] LogicalDeviceError),
    #[error("Swapchain creation error: {0}")]
    SwapchainInitError(#[from] SwapchainError),
    #[error("Textured triangle mesh pipeline creation error: {0}")]
    TTMPError(#[from] TTMPError),
    #[error("GPU allocator init error: {0}")]
    AllocatorInitError(gpu_allocator::AllocationError),
    #[error("Descriptor pool creation error: {0}")]
    DescriptorPoolError(#[from] DescriptorPoolError),
    #[error("Command pool error: {0}")]
    CommandPoolError(#[from] CommandPoolError),
    #[error("Command buffer error: {0}")]
    CommandBufferError(#[from] CommandBufferError),
}

#[derive(getset::Getters)]
pub struct Renderer {
    global_allocator: Arc<Mutex<Allocator>>,
    textures: HashMap<String, Texture>,
    tri_meshes: HashMap<String, TriMesh>,
    draw_cbs: Vec<CommandBuffer>,
    global_cp: Arc<CommandPool>,
    ttmp_attachments: Vec<TTMPAttachments>,
    ttmp_sets: Vec<TTMPSets>,
    ttmp: Arc<TTMP>,
    device: Arc<LogicalDevice>,
    instance: Arc<Instance>,
    #[get = "pub"]
    window: Arc<Window>,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Result<Self, RendererInitError> {
        let instance = Arc::new(Instance::new(window.clone())?);
        let device = Arc::new(LogicalDevice::new(instance.clone())?);
        let swapchain = Swapchain::new(device.clone())?;

        let ttmp = Arc::new(TTMP::new(device.clone())?);
        let global_allocator = device
            .make_allocator()
            .map(Mutex::new)
            .map(Arc::new)
            .map_err(RendererInitError::AllocatorInitError)?;

        let descriptor_pool = Arc::new(DescriptorPool::new(
            device.clone(),
            &[
                (vk::DescriptorType::SAMPLER, 10),
                (vk::DescriptorType::STORAGE_BUFFER, 50),
                (
                    vk::DescriptorType::SAMPLED_IMAGE,
                    10 * ttmp.max_textures() as u32,
                ),
            ],
            50,
        )?);

        let ttmp_attachments = (0..swapchain.images().len())
            .map(|_| {
                TTMPAttachments::new(ttmp.clone(), global_allocator.clone(), swapchain.extent())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ttmp_sets = (0..swapchain.images().len())
            .map(|_| {
                TTMPSets::new(
                    ttmp.clone(),
                    global_allocator.clone(),
                    descriptor_pool.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let global_cp = Arc::new(CommandPool::new(
            device.clone(),
            QueueType::Graphics,
            false,
        )?);

        let draw_cbs = CommandBuffer::new(global_cp.clone(), 3)?;

        Ok(Self {
            draw_cbs,
            global_cp,
            global_allocator,
            tri_meshes: HashMap::new(),
            textures: HashMap::new(),
            ttmp,
            ttmp_sets,
            ttmp_attachments,
            device,
            instance,
            window,
        })
    }

    pub fn add_mesh(&mut self, name: String, mesh: TriMesh) {
        self.tri_meshes.insert(name, mesh);
    }

    pub fn add_texture(&mut self, name: String, path: &Path) {
        let image_data = image::open(path).map_err();
        self.textures.insert(name, data);
    }

    pub fn draw(&mut self) {}
}
