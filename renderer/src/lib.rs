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
        textured_tri_mesh::{MaterialInfo, TTMP, TTMPAttachments, TTMPSets},
    },
    renderables::{camera::Camera, texture::Texture, tri_mesh::TriMesh},
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

        unsafe {
            self.device.sync2_device().cmd_pipeline_barrier2(
                command_buffer.command_buffer(),
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .image(image.image())
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                ]),
            );
        }

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

        unsafe {
            self.device.sync2_device().cmd_pipeline_barrier2(
                command_buffer.command_buffer(),
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .image(image.image())
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                ]),
            );
        }

        command_buffer.end()?;

        let fence = Fence::new(self.device.clone(), false)?;

        unsafe {
            device.queue_submit(
                self.device.graphics_queue(),
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer.command_buffer()])],
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
        let material_infos = (0..tex_list.len())
            .map(|i| MaterialInfo {
                sampler_id: 0 as u32,
                texture_id: i as u32,
                padding: [0, 0]
            })
            .collect::<Vec<_>>();

        let camera = Camera::new(glam::vec4(2.0, 2.0, 2.0, 0.0), glam::vec4(-1.0, -1.0, -1.0, 0.0), 90.0);

        self.ttmp_sets[draw_idx].update_ssbos(&self.dtp, &mesh_list, camera, &material_infos)?;
        self.ttmp_sets[draw_idx].update_textures(&tex_list);

        let draw_cb = &self.draw_cbs[draw_idx];

        let rendered_image = self.ttmp_attachments[draw_idx].color().image().image();
        let swapchain_image = self.swapchain.image_views()[draw_idx].image().image();

        // Record command buffer
        draw_cb.begin(true)?;

        unsafe {
            let im_barriers = if self.swapchain_initialized {
                vec![
                    vk::ImageMemoryBarrier2::default()
                        .image(rendered_image)
                        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                ]
            } else {
                let mut sw_ims = self
                    .swapchain
                    .image_views()
                    .iter()
                    .map(|swi| {
                        vk::ImageMemoryBarrier2::default()
                            .image(swi.image().image())
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .src_stage_mask(vk::PipelineStageFlags2::empty())
                            .src_access_mask(vk::AccessFlags2::empty())
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                            .subresource_range(vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1)
                            )
                    })
                    .collect::<Vec<_>>();
                let mut mr_ims = self
                    .ttmp_attachments
                    .iter()
                    .map(|mra| {
                        vk::ImageMemoryBarrier2::default()
                            .image(mra.color().image().image())
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::empty())
                            .src_access_mask(vk::AccessFlags2::empty())
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .subresource_range(vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1)
                            )
                    })
                    .collect::<Vec<_>>();
                let mut mr_ims_d = self
                    .ttmp_attachments
                    .iter()
                    .map(|mra| {
                        vk::ImageMemoryBarrier2::default()
                            .image(mra.depth().image().image())
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::empty())
                            .src_access_mask(vk::AccessFlags2::empty())
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                            .subresource_range(vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                                .layer_count(1)
                                .level_count(1)
                            )
                    })
                    .collect::<Vec<_>>();
                sw_ims.append(&mut mr_ims);
                sw_ims.append(&mut mr_ims_d);
                sw_ims
            };
            self.device.sync2_device().cmd_pipeline_barrier2(
                draw_cb.command_buffer(),
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&im_barriers),
            );
        }

        self.ttmp.render(
            draw_cb,
            &self.ttmp_sets[draw_idx],
            &self.ttmp_attachments[draw_idx],
        );

        let sw_res = self.swapchain.extent();
        unsafe {
            self.device.sync2_device().cmd_pipeline_barrier2(
                draw_cb.command_buffer(),
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&[
                        vk::ImageMemoryBarrier2::default()
                            .image(swapchain_image)
                            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                            .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .layer_count(1)
                                    .level_count(1),
                            ),
                        vk::ImageMemoryBarrier2::default()
                            .image(rendered_image)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .layer_count(1)
                                    .level_count(1),
                            ),
                    ]),
            );
            self.device.device().cmd_blit_image(
                draw_cb.command_buffer(),
                rendered_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: sw_res.width as _,
                            y: sw_res.height as _,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: sw_res.width as _,
                            y: sw_res.height as _,
                            z: 1,
                        },
                    ])],
                vk::Filter::NEAREST
            );

            self.device.sync2_device().cmd_pipeline_barrier2(
                draw_cb.command_buffer(),
                &vk::DependencyInfo::default()
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
                    .image_memory_barriers(&[vk::ImageMemoryBarrier2::default()
                        .image(swapchain_image)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        )]),
            );
        }

        draw_cb.end()?;

        let fence = Fence::new(self.device.clone(), false)?;

        unsafe {
            self.device.sync2_device().queue_submit2(
                self.device.graphics_queue(),
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&[
                        vk::CommandBufferSubmitInfo::default()
                            .command_buffer(draw_cb.command_buffer())
                    ])],
                fence.fence(),
            )?;
        }
        fence.wait(u64::MAX)?;
        fence.reset()?;

        self.swapchain.present(present_img_idx, &[])?;
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
            self.device.device().device_wait_idle();
        }
    }
}
