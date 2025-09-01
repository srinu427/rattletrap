pub mod pipelines;
pub mod renderables;
pub mod wrappers;

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Result as AnyResult;
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use hashbrown::HashMap;
use winit::window::Window;

use crate::{
    pipelines::{
        data_transfer::DTP,
        textured_tri_mesh::{TTMP, TTMPAttachments, TTMPSets},
    },
    renderables::{texture::Texture, tri_mesh::TriMesh},
    wrappers::{
        command_buffer::CommandBuffer,
        command_pool::CommandPool,
        descriptor_pool::DescriptorPool,
        image::Image,
        image_view::ImageView,
        instance::Instance,
        logical_device::{LogicalDevice, QueueType},
        swapchain::Swapchain,
    },
};

pub struct TTPMRenderable {
    mesh: String,
    texture: String,
}

#[derive(getset::Getters)]
pub struct Renderer {
    ttmp_renderables: HashMap<String, TTPMRenderable>,
    dtp: Arc<DTP>,
    global_allocator: Arc<Mutex<Allocator>>,
    textures: HashMap<String, Texture>,
    tri_meshes: HashMap<String, TriMesh>,
    draw_cbs: Vec<CommandBuffer>,
    global_cp: Arc<CommandPool>,
    ttmp_attachments: Vec<TTMPAttachments>,
    ttmp_sets: Vec<TTMPSets>,
    ttmp: Arc<TTMP>,
    swapchain: Arc<Swapchain>,
    device: Arc<LogicalDevice>,
    instance: Arc<Instance>,
    #[get = "pub"]
    window: Arc<Window>,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> AnyResult<Self> {
        let instance = Arc::new(Instance::new(window.clone())?);
        let device = Arc::new(LogicalDevice::new(instance.clone())?);
        let swapchain = Swapchain::new(device.clone())?;

        let ttmp = Arc::new(TTMP::new(device.clone())?);
        let global_allocator = device.make_allocator().map(Mutex::new).map(Arc::new)?;

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

        let ttmp_attachments = (0..swapchain.image_views().len())
            .map(|_| {
                TTMPAttachments::new(ttmp.clone(), global_allocator.clone(), swapchain.extent())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ttmp_sets = (0..swapchain.image_views().len())
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

        let dtp = Arc::new(DTP::new(device.clone(), global_allocator.clone())?);

        Ok(Self {
            ttmp_renderables: HashMap::new(),
            dtp,
            draw_cbs,
            global_cp,
            global_allocator,
            tri_meshes: HashMap::new(),
            textures: HashMap::new(),
            ttmp,
            ttmp_sets,
            ttmp_attachments,
            swapchain: Arc::new(swapchain),
            device,
            instance,
            window,
        })
    }

    pub fn add_mesh(&mut self, name: String, mesh: TriMesh) {
        self.tri_meshes.insert(name, mesh);
    }

    pub fn add_texture(&mut self, name: String, path: &Path) -> AnyResult<()> {
        let image_data = image::open(path)?;
        let extent = vk::Extent2D {
            width: image_data.width(),
            height: image_data.height(),
        };
        let mut image = Image::new_2d(
            self.device.clone(),
            vk::Format::R8G8B8A8_SRGB,
            extent,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        image.allocate_memory(self.global_allocator.clone(), true)?;
        self.dtp
            .transfer_data_to_image_2d(image_data.as_bytes(), &image)?;

        let image = Arc::new(image);

        let image_view = ImageView::new(
            image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .map(Arc::new)?;

        let texture = Texture::new(image_view);

        self.textures.insert(name, texture);

        Ok(())
    }

    pub fn add_ttpm_renderable(
        &mut self,
        name: String,
        mesh_name: String,
        texture_name: String,
    ) -> AnyResult<()> {
        let _ = self.tri_meshes.get(&mesh_name).ok_or_else(|| {
            anyhow::anyhow!("Mesh '{}' not found", mesh_name)
        })?;
        let _ = self.textures.get(&texture_name).ok_or_else(|| {
            anyhow::anyhow!("Texture '{}' not found", texture_name)
        })?;

        let renderable = TTPMRenderable {
            mesh: mesh_name,
            texture: texture_name,
        };

        self.ttmp_renderables.insert(name, renderable);

        Ok(())
    }

    pub fn draw(&mut self) {}
}
