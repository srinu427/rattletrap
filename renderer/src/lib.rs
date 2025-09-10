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
    renderables::{camera::Camera, texture::Texture, tri_mesh::TriMesh},
    wrappers::{
        command::{BarrierCommand, Command},
        command_buffer::CommandBuffer,
        command_pool::CommandPool,
        descriptor_pool::DescriptorPool,
        fence::Fence,
        image::Image,
        image_view::ImageView,
        instance::Instance,
        logical_device::{LogicalDevice, QueueType},
        semaphore::Semaphore,
        swapchain::Swapchain,
    },
};

pub struct TTPMRenderable {
    mesh: String,
    texture: String,
}

pub struct PerFrameData {
    draw_cb: CommandBuffer,
    draw_emit_sem: Semaphore,
    draw_fence: Fence,
    ttmp_set: TTMPSets,
    ttmp_attachments: TTMPAttachments,
}

impl PerFrameData {
    pub fn new(
        global_cp: Arc<CommandPool>,
        ttmp: Arc<TTMP>,
        global_allocator: Arc<Mutex<Allocator>>,
        descriptor_pool: Arc<DescriptorPool>,
        extent: vk::Extent2D,
    ) -> AnyResult<Self> {
        let draw_cb = CommandBuffer::new(global_cp.clone(), 1)?.remove(0);
        let draw_emit_sem = Semaphore::new(global_cp.device().clone())?;
        let draw_fence = Fence::new(global_cp.device().clone(), true)?;
        let ttmp_set = TTMPSets::new(ttmp.clone(), global_allocator.clone(), descriptor_pool)?;
        let ttmp_attachments = TTMPAttachments::new(ttmp, global_allocator, extent)?;

        Ok(Self {
            draw_cb,
            draw_emit_sem,
            draw_fence,
            ttmp_set,
            ttmp_attachments,
        })
    }

    pub fn resize(
        &mut self,
        allocator: Arc<Mutex<Allocator>>,
        extent: vk::Extent2D,
    ) -> AnyResult<()> {
        let new_ttmp_attachments =
            TTMPAttachments::new(self.ttmp_attachments.ttmp().clone(), allocator, extent)?;
        self.ttmp_attachments = new_ttmp_attachments;
        Ok(())
    }
}

#[derive(getset::Getters)]
pub struct Renderer {
    per_frame_datas: Vec<PerFrameData>,
    next_image_acquire_fence: Arc<Fence>,
    ttmp_renderables: HashMap<String, TTPMRenderable>,
    dtp: Arc<DTP>,
    global_allocator: Arc<Mutex<Allocator>>,
    textures: HashMap<String, Texture>,
    tri_meshes: HashMap<String, TriMesh>,
    global_cp: Arc<CommandPool>,
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

        let global_cp = Arc::new(CommandPool::new(
            device.clone(),
            QueueType::Graphics,
            false,
        )?);

        let dtp = Arc::new(DTP::new(device.clone(), global_allocator.clone())?);

        let next_image_acquire_fence = Arc::new(Fence::new(device.clone(), false)?);

        let per_frame_datas = (0..swapchain.image_views().len())
            .map(|_| {
                PerFrameData::new(
                    global_cp.clone(),
                    ttmp.clone(),
                    global_allocator.clone(),
                    descriptor_pool.clone(),
                    swapchain.extent(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            per_frame_datas,
            next_image_acquire_fence,
            ttmp_renderables: HashMap::new(),
            dtp,
            global_cp,
            global_allocator,
            tri_meshes: HashMap::new(),
            textures: HashMap::new(),
            ttmp,
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
        let device = self.device.device();
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

        let command_buffer = self.dtp.create_temp_command_buffer()?;

        command_buffer.begin(true)?;

        Command::Barrier(BarrierCommand::Image2d {
            image: &image,
            old_layout: vk::ImageLayout::UNDEFINED,
            old_stage: vk::PipelineStageFlags2::NONE,
            old_access: vk::AccessFlags2::empty(),
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_stage: vk::PipelineStageFlags2::TRANSFER,
            new_access: vk::AccessFlags2::TRANSFER_WRITE,
            aspect_mask: vk::ImageAspectFlags::COLOR,
        })
        .record(&command_buffer);

        let stage_buffer = self.dtp.do_transfers_custom(
            vec![DTPInput::CopyToImage {
                data: image_data.as_bytes(),
                image: &image,
                subresource_layers: vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            }],
            &command_buffer,
        )?;

        Command::Barrier(BarrierCommand::Image2d {
            image: &image,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            old_stage: vk::PipelineStageFlags2::TRANSFER,
            old_access: vk::AccessFlags2::TRANSFER_WRITE,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            new_access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            aspect_mask: vk::ImageAspectFlags::COLOR,
        })
        .record(&command_buffer);

        command_buffer.end()?;

        let fence = Fence::new(self.device.clone(), false)?;

        unsafe {
            self.device.sync2_device().queue_submit2(
                self.device.graphics_queue(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[
                        vk::CommandBufferSubmitInfo::default()
                            .command_buffer(command_buffer.command_buffer())
                        ])
                ],
                fence.fence(),
            )?;
        }
        fence.wait(u64::MAX)?;

        drop(stage_buffer);

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
        let (present_img_idx, refreshed) = self
            .swapchain
            .acquire_image(&self.next_image_acquire_fence)?;

        if refreshed || !self.swapchain_initialized {
            self.swapchain_initialized = false;
        }
        let draw_idx = present_img_idx as usize;

        if self.swapchain.extent() != self.per_frame_datas[draw_idx].ttmp_attachments.extent() {
            self.per_frame_datas[draw_idx]
                .resize(self.global_allocator.clone(), self.swapchain.extent())?;
        }

        let mut meshes_per_material = HashMap::new();
        for (_, renderable) in &self.ttmp_renderables {
            meshes_per_material
                .entry(renderable.texture.clone())
                .or_insert(vec![])
                .push(renderable.mesh.clone());
        }

        let mut tex_list = vec![];
        let mut mesh_list = vec![];

        for (tex_name, mesh_names) in meshes_per_material.iter() {
            let texture = self
                .textures
                .get(tex_name)
                .ok_or_else(|| anyhow::anyhow!("Texture '{}' not found", tex_name))?;
            tex_list.push(texture);
            let tex_id = (tex_list.len() - 1) as u32;
            for mesh_name in mesh_names {
                let mut mesh = self
                    .tri_meshes
                    .get(mesh_name)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("mesh '{}' not found", mesh_name))?;

                mesh.write_obj_id(tex_id);
                mesh_list.push(mesh);
            }
        }

        let camera = Camera::new(
            glam::vec4(1.0, 1.0, 1.0, 0.0),
            glam::vec4(-1.0, -1.0, -1.0, 0.0),
            70.0,
        );

        self.per_frame_datas[draw_idx]
            .ttmp_set
            .update_ssbos(&self.dtp, &mesh_list, camera)?;
        self.per_frame_datas[draw_idx]
            .ttmp_set
            .update_textures(&tex_list);

        let draw_cb = &self.per_frame_datas[draw_idx].draw_cb;

        let mut commands = vec![];

        // Record command buffer
        self.per_frame_datas[draw_idx]
            .draw_fence
            .wait(u64::MAX)?;
        self.per_frame_datas[draw_idx].draw_fence.reset()?;
        
        draw_cb.reset()?;
        draw_cb.begin(true)?;

        if !self.swapchain_initialized {
            let mut sw_ims = self
                .swapchain
                .image_views()
                .iter()
                .map(|swi| {
                    Command::Barrier(BarrierCommand::Image2d {
                        image: swi.image(),
                        old_layout: vk::ImageLayout::UNDEFINED,
                        old_stage: vk::PipelineStageFlags2::NONE,
                        old_access: vk::AccessFlags2::empty(),
                        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                        new_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                        new_access: vk::AccessFlags2::NONE,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    })
                })
                .collect::<Vec<_>>();
            let mut mr_ims = (0..self.per_frame_datas.len())
                .map(|i| {
                    Command::Barrier(BarrierCommand::Image2d {
                        image: self.per_frame_datas[i].ttmp_attachments.color().image(),
                        old_layout: vk::ImageLayout::UNDEFINED,
                        old_stage: vk::PipelineStageFlags2::NONE,
                        old_access: vk::AccessFlags2::empty(),
                        new_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
                        new_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        new_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    })
                })
                .collect::<Vec<_>>();
            let mut mr_ims_d = (0..self.per_frame_datas.len())
                .map(|i| {
                    Command::Barrier(BarrierCommand::Image2d {
                        image: self.per_frame_datas[i].ttmp_attachments.depth().image(),
                        old_layout: vk::ImageLayout::UNDEFINED,
                        old_stage: vk::PipelineStageFlags2::NONE,
                        old_access: vk::AccessFlags2::empty(),
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        new_stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                        new_access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                        aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    })
                })
                .collect::<Vec<_>>();
            commands.append(&mut sw_ims);
            commands.append(&mut mr_ims);
            commands.append(&mut mr_ims_d);
        };

        for command in &commands {
            command.record(draw_cb);
        }

        self.ttmp.render(
            draw_cb,
            &self.per_frame_datas[draw_idx].ttmp_set,
            &self.per_frame_datas[draw_idx].ttmp_attachments,
        );

        let commands = vec![
            Command::Barrier(BarrierCommand::Image2d {
                image: &self.per_frame_datas[draw_idx]
                    .ttmp_attachments
                    .color()
                    .image(),
                old_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
                old_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                old_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_stage: vk::PipelineStageFlags2::BLIT,
                new_access: vk::AccessFlags2::TRANSFER_READ,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            }),
            Command::Barrier(BarrierCommand::Image2d {
                image: self.swapchain.image_views()[draw_idx].image(),
                old_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                old_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                old_access: vk::AccessFlags2::NONE,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_stage: vk::PipelineStageFlags2::BLIT,
                new_access: vk::AccessFlags2::TRANSFER_WRITE,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            }),
            Command::BlitImage2dFull {
                src: &self.per_frame_datas[draw_idx]
                    .ttmp_attachments
                    .color()
                    .image(),
                dst: self.swapchain.image_views()[draw_idx].image(),
                filter: vk::Filter::NEAREST,
            },
            Command::Barrier(BarrierCommand::Image2d {
                image: self.swapchain.image_views()[draw_idx].image(),
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                old_stage: vk::PipelineStageFlags2::BLIT,
                old_access: vk::AccessFlags2::TRANSFER_WRITE,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                new_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                new_access: vk::AccessFlags2::NONE,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            }),
            Command::Barrier(BarrierCommand::Image2d {
                image: &self.per_frame_datas[draw_idx]
                    .ttmp_attachments
                    .color()
                    .image(),
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                old_stage: vk::PipelineStageFlags2::BLIT,
                old_access: vk::AccessFlags2::TRANSFER_READ,
                new_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
                new_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                new_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            }),
        ];

        for command in &commands {
            command.record(draw_cb);
        }

        draw_cb.end()?;

        self.next_image_acquire_fence.wait(u64::MAX)?;
        self.next_image_acquire_fence.reset()?;

        unsafe {
            self.device.sync2_device().queue_submit2(
                self.device.graphics_queue(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                        .command_buffer(draw_cb.command_buffer())])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.per_frame_datas[draw_idx].draw_emit_sem.semaphore())
                        .stage_mask(vk::PipelineStageFlags2::BLIT)])],
               self.per_frame_datas[draw_idx].draw_fence.fence(),
            )?;
        }

        self.swapchain_initialized = true;

        self.swapchain.present(present_img_idx, &[&self.per_frame_datas[draw_idx].draw_emit_sem])?;
        Ok(())
    }

    pub fn refresh_resolution(&mut self) -> AnyResult<()> {
        self.swapchain.refresh_resolution()?;
        self.swapchain_initialized = false;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device().device_wait_idle().ok();
        }
    }
}
