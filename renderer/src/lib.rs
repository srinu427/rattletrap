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
        image::{Image, ImageAccess},
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
    ) -> AnyResult<(Self, Vec<Command>)> {
        let draw_cb = CommandBuffer::new(global_cp.clone(), 1)?.remove(0);
        let draw_emit_sem = Semaphore::new(global_cp.device().clone())?;
        let draw_fence = Fence::new(global_cp.device().clone(), false)?;
        let ttmp_set = TTMPSets::new(ttmp.clone(), global_allocator.clone(), descriptor_pool)?;
        let (ttmp_attachments, commands) = TTMPAttachments::new(ttmp, global_allocator, extent)?;

        Ok((Self {
            draw_cb,
            draw_emit_sem,
            draw_fence,
            ttmp_set,
            ttmp_attachments,
        },
        commands))
    }

    pub fn resize(
        &mut self,
        allocator: Arc<Mutex<Allocator>>,
        extent: vk::Extent2D,
    ) -> AnyResult<Vec<Command>> {
        let (new_ttmp_attachments, init_cmds) =
            TTMPAttachments::new(self.ttmp_attachments.ttmp().clone(), allocator, extent)?;
        self.ttmp_attachments = new_ttmp_attachments;
        Ok(init_cmds)
    }
}

#[derive(getset::Getters)]
pub struct Renderer {
    per_frame_datas: Vec<PerFrameData>,
    ttmp_renderables: HashMap<String, TTPMRenderable>,
    dtp: Arc<DTP>,
    global_allocator: Arc<Mutex<Allocator>>,
    textures: HashMap<String, Texture>,
    tri_meshes: HashMap<String, TriMesh>,
    global_cp: Arc<CommandPool>,
    ttmp: Arc<TTMP>,
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
        let (swapchain, sw_init_commands) = Swapchain::new(device.clone())?;

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

        let (per_frame_datas, pfd_commands): (Vec<_>, Vec<_>) = per_frame_datas
            .into_iter()
            .unzip();

        let pfd_commands = pfd_commands.into_iter().flatten().collect::<Vec<_>>();

        let init_commands = sw_init_commands
            .into_iter()
            .chain(pfd_commands.into_iter())
            .collect::<Vec<_>>();

        per_frame_datas[0].draw_cb.record_commands(&init_commands, true)?;
        per_frame_datas[0].draw_cb.submit(&[], &[], Some(&per_frame_datas[0].draw_fence))?;
        per_frame_datas[0].draw_fence.wait(u64::MAX)?;
        per_frame_datas[0].draw_fence.reset()?;

        Ok(Self {
            per_frame_datas,
            ttmp_renderables: HashMap::new(),
            dtp,
            global_cp,
            global_allocator,
            tri_meshes: HashMap::new(),
            textures: HashMap::new(),
            ttmp,
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
            vec![ImageAccess::TransferDst, ImageAccess::TransferSrc, ImageAccess::ShaderRead],
        )?;
        image.allocate_memory(self.global_allocator.clone(), true)?;
        let image = Arc::new(image);

        let mut commands = vec![
            Command::Barrier(BarrierCommand::new_image_2d_barrier(
                &image,
                ImageAccess::Undefined,
                ImageAccess::TransferDst,
            )),
        ];

        let (stage_buffer, upload_cmds) = self.dtp.do_transfers_custom(
            vec![DTPInput::CopyToImage {
                data: image_data.as_bytes(),
                image: &image,
                subresource_layers: image.all_subresource_layers(0),
            }],
        )?;

        commands.extend(upload_cmds);

        commands.push(Command::Barrier(BarrierCommand::new_image_2d_barrier(
            &image,
            ImageAccess::TransferDst,
            ImageAccess::ShaderRead,
        )));

        let command_buffer = self.dtp.create_temp_command_buffer()?;

        command_buffer.record_commands(&commands, true)?;

        let fence = Fence::new(self.device.clone(), false)?;

        command_buffer.submit(&[], &[], Some(&fence))?;

        fence.wait(u64::MAX)?;

        drop(stage_buffer);

        let image_view = ImageView::new(
            image.clone(),
            vk::ImageViewType::TYPE_2D,
            image.full_subresource_range(),
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
        let (present_img_idx, init_cmds) = self.swapchain.acquire_image()?;

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

        // Record command buffer
        let (update_stage_buffer, update_cmds) = self.per_frame_datas[draw_idx]
            .ttmp_set
            .update_ssbos(&self.dtp, &mesh_list, camera)?;
        self.per_frame_datas[draw_idx]
            .ttmp_set
            .update_textures(&tex_list);

        let draw_cb = &self.per_frame_datas[draw_idx].draw_cb;

        let mut commands = vec![];

        let ttpm_cmds = self.ttmp.render(
            &self.per_frame_datas[draw_idx].ttmp_set,
            &self.per_frame_datas[draw_idx].ttmp_attachments,
        );

        let post_sync_commands = vec![
            Command::Barrier(BarrierCommand::new_image_2d_barrier(
                self.swapchain.image_views()[draw_idx].image(),
                ImageAccess::Present,
                ImageAccess::TransferDst,
            )),
            // Command::blit_full_image(
            //     self.per_frame_datas[draw_idx]
            //         .ttmp_attachments
            //         .color()
            //         .image(),
            //     self.swapchain.image_views()[draw_idx].image(),
            //     vk::Filter::NEAREST,
            // ),
            Command::blit_full_image(
                self.textures["default"].albedo().image(),
                self.swapchain.image_views()[draw_idx].image(),
                vk::Filter::NEAREST,
            ),
            Command::Barrier(BarrierCommand::new_image_2d_barrier(
                self.swapchain.image_views()[draw_idx].image(),
                ImageAccess::TransferDst,
                ImageAccess::Present,
            )),
        ];

        commands.extend(init_cmds);
        commands.extend(update_cmds);

        draw_cb.reset()?;
        draw_cb.record_commands(&commands, false)?;

        draw_cb.submit(
            &[],
            &[],
            Some(&self.per_frame_datas[draw_idx].draw_fence)
        )?;

        self.per_frame_datas[draw_idx].draw_fence.wait(u64::MAX)?;
        self.per_frame_datas[draw_idx].draw_fence.reset()?;

        commands.clear();

        // commands.extend(ttpm_cmds);
        commands.extend(post_sync_commands);

        draw_cb.reset()?;
        draw_cb.record_commands(&commands, false)?;

        draw_cb.submit(
            &[],
            &[(&self.per_frame_datas[draw_idx].draw_emit_sem, vk::PipelineStageFlags2::BOTTOM_OF_PIPE)],
            Some(&self.per_frame_datas[draw_idx].draw_fence)
        )?;

        self.per_frame_datas[draw_idx].draw_fence.wait(u64::MAX)?;
        self.per_frame_datas[draw_idx].draw_fence.reset()?;
        drop(update_stage_buffer);

        self.swapchain.present(
            present_img_idx,
            &[&self.per_frame_datas[draw_idx].draw_emit_sem],
        )?;
        Ok(())
    }

    pub fn refresh_resolution(&mut self) -> AnyResult<()> {
        self.swapchain.refresh_resolution()?;
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
