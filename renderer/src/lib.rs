pub mod pipelines;
pub mod renderables;
pub mod wrappers;

use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use hashbrown::HashMap;
use thiserror::Error;
use winit::window::Window;

use crate::{
    pipelines::textured_tri_mesh::{
        TTMPAttachments, TTMPAttachmentsError, TTMPInitError, TTMPSets, TTMPSetsError, TTMP
    },
    renderables::tri_mesh::TriMesh,
    wrappers::{
        descriptor_pool::DescriptorPool,
        instance::{Instance, InstanceError},
        logical_device::{LogicalDevice, LogicalDeviceInitError},
        swapchain::{Swapchain, SwapchainError},
    },
};

#[derive(Debug, Error)]
pub enum RendererInitError {
    #[error("Instance error: {0}")]
    InstanceError(#[from] InstanceError),
    #[error("Logical device creation error: {0}")]
    LogicalDeviceInitError(#[from] LogicalDeviceInitError),
    #[error("Swapchain creation error: {0}")]
    SwapchainInitError(#[from] SwapchainError),
    #[error("Textured triangle mesh pipeline creation error: {0}")]
    TTMPInitError(#[from] TTMPInitError),
    #[error("GPU allocator init error: {0}")]
    AllocatorInitError(gpu_allocator::AllocationError),
    #[error("Descriptor pool creation error: {0}")]
    DescriptorPoolError(vk::Result),
    #[error("TTMP set creation error: {0}")]
    TTMPSetsError(#[from] TTMPSetsError),
    #[error("TTMP attachments creation error: {0}")]
    TTMPAttachmentsError(#[from] TTMPAttachmentsError),
}

#[derive(getset::Getters)]
pub struct Renderer {
    global_allocator: Arc<Mutex<Allocator>>,
    tri_meshes: HashMap<String, TriMesh>,
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

        let descriptor_pool = Arc::new(
            DescriptorPool::new(
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
            )
            .map_err(RendererInitError::DescriptorPoolError)?,
        );

        let ttmp_attachments = swapchain
            .images()
            .iter()
            .map(|image| {
                TTMPAttachments::new(ttmp.clone(), global_allocator.clone(), swapchain.extent())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ttmp_sets = swapchain
            .images()
            .iter()
            .map(|_| TTMPSets::new(ttmp.clone(), global_allocator.clone(), descriptor_pool.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            global_allocator,
            tri_meshes: HashMap::new(),
            ttmp,
            ttmp_sets,
            ttmp_attachments,
            device,
            instance,
            window,
        })
    }
}
