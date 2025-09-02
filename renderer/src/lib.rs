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
        data_transfer::{DTP, DTPInput},
        textured_tri_mesh::{TTMP, TTMPAttachments, TTMPSets},
    },
    renderables::{texture::Texture, tri_mesh::TriMesh},
    wrappers::{
        command_buffer::CommandBuffer,
        command_pool::CommandPool,
        descriptor_pool::DescriptorPool,
        fence::Fence,
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
    next_image_acquire_fence: Arc<Fence>,
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
    swapchain_initialized: bool,
    swapchain: Swapchain,
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

        let next_image_acquire_fence = Arc::new(Fence::new(device.clone(), false)?);

        Ok(Self {
            next_image_acquire_fence,
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
            swapchain_initialized: false,
            swapchain,
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
        let image = Arc::new(image);

        self.dtp.do_transfers(vec![DTPInput::CopyToImage {
            data: image_data.as_bytes(),
            image: &image,
            subresource_layers: vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        }])?;

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
        let _ = self
            .tri_meshes
            .get(&mesh_name)
            .ok_or_else(|| anyhow::anyhow!("Mesh '{}' not found", mesh_name))?;
        let _ = self
            .textures
            .get(&texture_name)
            .ok_or_else(|| anyhow::anyhow!("Texture '{}' not found", texture_name))?;

        let renderable = TTPMRenderable {
            mesh: mesh_name,
            texture: texture_name,
        };

        self.ttmp_renderables.insert(name, renderable);

        Ok(())
    }

    pub fn draw(&mut self) -> AnyResult<()> {
        // Aquire next image from swapchain
        let present_img_idx = self
            .swapchain
            .acquire_image(&self.next_image_acquire_fence)?;
        self.next_image_acquire_fence.wait(u64::MAX)?;
        self.next_image_acquire_fence.reset()?;

        let draw_idx = present_img_idx as usize;

        if self.swapchain.extent() != self.ttmp_attachments[draw_idx].extent() {
            let new_ttmp_attachments = (0..self.swapchain.image_views().len())
                .map(|_| {
                    TTMPAttachments::new(
                        self.ttmp.clone(),
                        self.global_allocator.clone(),
                        self.swapchain.extent(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.ttmp_attachments = new_ttmp_attachments;
        }
        let ttmp_attachment = &self.ttmp_attachments[draw_idx];

        let ttmp_sets = &self.ttmp_sets[draw_idx];

        let mut meshes_per_material = HashMap::new();
        for (_, renderable) in &self.ttmp_renderables {
            meshes_per_material
                .entry(renderable.texture.clone())
                .or_insert(vec![])
                .push(renderable.mesh.clone());
        }

        let tex_list = vec![];
        let mesh_list = vec![];

        let draw_cb = &self.draw_cbs[draw_idx];
        // Record command buffer
        draw_cb.begin(true)?;

        draw_cb.end()?;

        self.swapchain.present(present_img_idx, &[])?;
        Ok(())
    }

    pub fn refresh_resolution(&mut self) -> AnyResult<()> {
        self.swapchain.refresh_resolution()?;
        Ok(())
    }
}
