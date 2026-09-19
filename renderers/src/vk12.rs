use std::fs;
use std::sync::Arc;

use ash::vk;
use bytemuck::NoUninit;
use hashbrown::HashMap;
use naga::back::spv;
use naga::front::glsl;
use naga::valid;

use winit::window::Window;

mod device;
mod resource;
mod swapchain;

use crate::{
    Camera3d, Light, Material, Mesh, MeshVertex, Scene,
    vk12::{
        device::{GpuCommandRecorder, GpuCtx},
        resource::{
            GpuBuffer, GpuDsl, GpuImage, GpuImageAccess, GpuImageViewInfo, GpuVecData,
            is_depth_stencil,
        },
        swapchain::GpuSwapchain,
    },
};

fn convert_glsl_to_spv(glsl_source: &str, stage: naga::ShaderStage) -> anyhow::Result<Vec<u32>> {
    let options = glsl::Options {
        stage,
        defines: Default::default(),
    };

    let mut frontend = glsl::Frontend::default();
    let module = frontend.parse(&options, glsl_source)?;

    let mut validator =
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all());
    let module_info = validator.validate(&module)?;

    let writer_flags = spv::WriterFlags::empty();
    let mut writer_options = spv::Options::default();
    writer_options.flags = writer_flags;

    let mut writer = spv::Writer::new(&writer_options)?;
    let mut spv_words = vec![];
    writer.write(&module, &module_info, None, &None, &mut spv_words)?;

    Ok(spv_words)
}

fn load_glsl(
    ctx: &GpuCtx,
    code: &str,
    stage: naga::ShaderStage,
) -> anyhow::Result<vk::ShaderModule> {
    let spv_words = convert_glsl_to_spv(code, stage)?;
    let shader_mod = unsafe {
        ctx.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&spv_words),
            None,
        )?
    };
    Ok(shader_mod)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct MeshPipelinePushConstant {
    obj_id: u32,
    material_id: u32,
}

struct RenderPipelineVk12 {
    render_pass: vk::RenderPass,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsls: Vec<GpuDsl>,
    framebuffers: HashMap<Vec<vk::ImageView>, vk::Framebuffer>,
}

impl RenderPipelineVk12 {
    fn new_mesh_pipeline(ctx: &mut GpuCtx, format: vk::Format) -> anyhow::Result<Self> {
        let render_pass = unsafe {
            ctx.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[
                        vk::AttachmentDescription::default()
                            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .format(format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE),
                        vk::AttachmentDescription::default()
                            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .format(vk::Format::D32_SFLOAT)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE),
                    ])
                    .subpasses(&[vk::SubpassDescription::default()
                        .color_attachments(&[vk::AttachmentReference::default()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
                        .depth_stencil_attachment(
                            &vk::AttachmentReference::default()
                                .attachment(1)
                                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                        )
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]),
                None,
            )?
        };
        let dsls = vec![
            GpuDsl::new(ctx, vec![(vk::DescriptorType::UNIFORM_BUFFER, 1)])?,
            GpuDsl::new(ctx, vec![(vk::DescriptorType::STORAGE_BUFFER, 1)])?,
            GpuDsl::new(ctx, vec![(vk::DescriptorType::STORAGE_BUFFER, 1)])?,
        ];
        let layout = unsafe {
            ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[dsls[0].handle, dsls[1].handle, dsls[2].handle])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .size(size_of::<MeshPipelinePushConstant>() as _)
                        .stage_flags(vk::ShaderStageFlags::ALL)]),
                None,
            )?
        };
        let vert_shader = load_glsl(
            ctx,
            include_str!("vk12_shaders/mesh.vert"),
            naga::ShaderStage::Vertex,
        )?;
        let frag_shader = load_glsl(
            ctx,
            include_str!("vk12_shaders/mesh.frag"),
            naga::ShaderStage::Fragment,
        )?;
        let pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::default(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .layout(layout)
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .cull_mode(vk::CullModeFlags::NONE)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0)
                                .polygon_mode(vk::PolygonMode::FILL),
                        )
                        .render_pass(render_pass)
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(vert_shader)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::VERTEX),
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(frag_shader)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::FRAGMENT),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .location(0)
                                        .offset(0),
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .location(1)
                                        .offset(12),
                                ])
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .input_rate(vk::VertexInputRate::VERTEX)
                                        .stride(size_of::<MeshVertex>() as _),
                                ]),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe {
            ctx.device.destroy_shader_module(vert_shader, None);
            ctx.device.destroy_shader_module(frag_shader, None);
        }
        Ok(Self {
            render_pass,
            handle: pipeline,
            layout,
            dsls,
            framebuffers: Default::default(),
        })
    }

    fn new_smap_pipeline(ctx: &mut GpuCtx) -> anyhow::Result<Self> {
        let render_pass = unsafe {
            ctx.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[vk::AttachmentDescription::default()
                        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .format(vk::Format::D32_SFLOAT)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)])
                    .subpasses(&[vk::SubpassDescription::default()
                        .depth_stencil_attachment(
                            &vk::AttachmentReference::default()
                                .attachment(0)
                                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                        )
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]),
                None,
            )?
        };
        let dsls = vec![
            GpuDsl::new(ctx, vec![(vk::DescriptorType::UNIFORM_BUFFER, 1)])?,
            GpuDsl::new(ctx, vec![(vk::DescriptorType::STORAGE_BUFFER, 1)])?,
        ];
        let layout = unsafe {
            ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[dsls[0].handle, dsls[1].handle])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .size(size_of::<MeshPipelinePushConstant>() as _)
                        .stage_flags(vk::ShaderStageFlags::ALL)]),
                None,
            )?
        };
        let vert_shader = load_glsl(
            ctx,
            include_str!("vk12_shaders/smap.vert"),
            naga::ShaderStage::Vertex,
        )?;
        let frag_shader = load_glsl(
            ctx,
            include_str!("vk12_shaders/smap.frag"),
            naga::ShaderStage::Fragment,
        )?;
        let pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::default(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .layout(layout)
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .cull_mode(vk::CullModeFlags::NONE)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0)
                                .polygon_mode(vk::PolygonMode::FILL),
                        )
                        .render_pass(render_pass)
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(vert_shader)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::VERTEX),
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(frag_shader)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::FRAGMENT),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .location(0)
                                        .offset(0),
                                ])
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .input_rate(vk::VertexInputRate::VERTEX)
                                        .stride(size_of::<MeshVertex>() as _),
                                ]),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe {
            ctx.device.destroy_shader_module(vert_shader, None);
            ctx.device.destroy_shader_module(frag_shader, None);
        }
        Ok(Self {
            render_pass,
            handle: pipeline,
            layout,
            dsls,
            framebuffers: Default::default(),
        })
    }

    fn get_frame_buffer(
        &mut self,
        ctx: &GpuCtx,
        mut images: Vec<&mut GpuImage>,
    ) -> anyhow::Result<vk::Framebuffer> {
        let mut views = vec![];
        for image in &mut images {
            let view = image.get_view(
                &ctx,
                GpuImageViewInfo {
                    type_: vk::ImageViewType::TYPE_2D,
                    layer_range: 0..1,
                    level_range: 0..1,
                },
            )?;
            views.push(view);
        }
        let res = images[0].res;
        let fb = self.framebuffers.get(&views).cloned();
        let fb = match fb {
            Some(t) => t,
            None => {
                let fb = unsafe {
                    ctx.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .attachments(&views)
                            .width(res.0)
                            .height(res.1)
                            .layers(1)
                            .render_pass(self.render_pass),
                        None,
                    )?
                };
                if self.framebuffers.len() > 128 {
                    unsafe {
                        for (_, fb) in self.framebuffers.drain() {
                            ctx.device.destroy_framebuffer(fb, None);
                        }
                    }
                }
                self.framebuffers.insert(views, fb);
                fb
            }
        };
        Ok(fb)
    }

    fn start(
        &mut self,
        ctx: &mut GpuCtx,
        cb: vk::CommandBuffer,
        mut attachments: Vec<&mut GpuImage>,
    ) -> anyhow::Result<()> {
        for attachment in &mut attachments {
            let (is_depth, _) = is_depth_stencil(attachment.format);
            let new_access = if is_depth {
                GpuImageAccess {
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                }
            } else {
                GpuImageAccess {
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                }
            };
            attachment.transition(ctx, cb, new_access);
        }
        let res = attachments[0].res;
        let framebuffer = self.get_frame_buffer(ctx, attachments)?;
        unsafe {
            ctx.device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue::default(),
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0),
                        },
                    ])
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                        width: res.0,
                        height: res.1,
                    }))
                    .render_pass(self.render_pass),
                vk::SubpassContents::INLINE,
            );
            ctx.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.handle);
            ctx.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: res.1 as _,
                    width: res.0 as _,
                    height: -(res.1 as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            ctx.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(vk::Extent2D {
                        width: res.0,
                        height: res.1,
                    })],
            );
        }
        Ok(())
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            for (_, fb) in self.framebuffers.drain() {
                ctx.device.destroy_framebuffer(fb, None);
            }
            ctx.device.destroy_pipeline(self.handle, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device.destroy_render_pass(self.render_pass, None);
        }
        for mut dsl in self.dsls.drain(..) {
            dsl.destroy(ctx);
        }
    }
}

struct GpuMaterialMgr {
    materials: HashMap<String, u32>,
    buffer: GpuVecData<Material>,
    dset: vk::DescriptorSet,
}

impl GpuMaterialMgr {
    fn new(ctx: &mut GpuCtx, dpool: &mut GpuDsl) -> anyhow::Result<Self> {
        let buffer = GpuVecData::new(ctx)?;
        let dset = dpool.get_set(ctx)?;
        let out = Self {
            materials: Default::default(),
            buffer,
            dset,
        };
        out.sync_dset(ctx);
        Ok(out)
    }

    fn sync_dset(&self, ctx: &GpuCtx) {
        unsafe {
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(self.buffer.buffer.handle)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(self.dset)],
                &[],
            );
        }
    }

    fn add_material(
        &mut self,
        ctx: &mut GpuCtx,
        name: &str,
        material: Material,
        cr: &mut GpuCommandRecorder,
    ) -> anyhow::Result<()> {
        if self.materials.contains_key(name) {
            return Ok(());
        }
        let insert_idx = self.buffer.len;
        let old_buffer_handle = self.buffer.buffer.handle;
        self.buffer.push(ctx, material, cr)?;
        let new_buffer_handle = self.buffer.buffer.handle;
        if old_buffer_handle != new_buffer_handle {
            self.sync_dset(ctx);
        }
        self.materials.insert(name.to_string(), insert_idx);
        Ok(())
    }

    fn clear(&mut self) {
        self.materials.clear();
        self.buffer.clear();
    }

    fn destroy(&mut self, ctx: &mut GpuCtx, dpool: &mut GpuDsl) {
        self.buffer.destroy(ctx);
        dpool.reclaim(self.dset);
    }
}

struct GpuLoadedMesh {
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    draw_count: u32,
}

impl GpuLoadedMesh {
    fn new(ctx: &mut GpuCtx, cr: &mut GpuCommandRecorder, mesh: Mesh) -> anyhow::Result<Self> {
        let vb_size = (mesh.vertices.len() * size_of::<MeshVertex>()) as u64;
        let ib_size = (mesh.indices.len() * size_of::<u16>()) as u64;
        let mut vbo =
            GpuBuffer::new_gpu_local_ro(ctx, vb_size, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let mut ibo =
            GpuBuffer::new_gpu_local_ro(ctx, ib_size, vk::BufferUsageFlags::INDEX_BUFFER)?;
        let draw_count = mesh.indices.len() as u32;
        cr.write_to_gpu_buffer(ctx, &mut vbo, 0, bytemuck::cast_slice(&mesh.vertices))?;
        cr.write_to_gpu_buffer(ctx, &mut ibo, 0, bytemuck::cast_slice(&mesh.indices))?;
        Ok(Self {
            vbo,
            ibo,
            draw_count,
        })
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        self.vbo.destroy(ctx);
        self.ibo.destroy(ctx);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct GpuMeshInfo {
    transform: glam::Mat4,
}

struct GpuMeshMgr {
    meshes: HashMap<String, GpuLoadedMesh>,
    info_buffer: GpuBuffer,
    dset: vk::DescriptorSet,
}

impl GpuMeshMgr {
    fn new(ctx: &mut GpuCtx, dpool: &mut GpuDsl) -> anyhow::Result<Self> {
        let info_buffer = GpuBuffer::new_gpu_local_ro(
            ctx,
            128 * size_of::<GpuMeshInfo>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let dset = dpool.get_set(ctx)?;
        unsafe {
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(info_buffer.handle)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(dset)],
                &[],
            );
        }
        Ok(Self {
            meshes: Default::default(),
            info_buffer,
            dset,
        })
    }

    fn add_mesh(
        &mut self,
        ctx: &mut GpuCtx,
        name: &str,
        mesh: Mesh,
        cr: &mut GpuCommandRecorder,
    ) -> anyhow::Result<()> {
        let gpu_mesh = GpuLoadedMesh::new(ctx, cr, mesh)?;
        self.meshes.insert(name.to_string(), gpu_mesh);
        Ok(())
    }

    fn write_data(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        data: &[GpuMeshInfo],
    ) -> anyhow::Result<()> {
        let needed_size = (data.len() * size_of::<GpuMeshInfo>()) as u64;
        if self.info_buffer.len < needed_size {
            let info_buffer = GpuBuffer::new_gpu_local_ro(
                ctx,
                needed_size.next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            self.info_buffer.destroy(ctx);
            self.info_buffer = info_buffer;
            unsafe {
                ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(self.info_buffer.handle)
                            .range(vk::WHOLE_SIZE)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_set(self.dset)],
                    &[],
                );
            }
        }
        cr.write_to_gpu_buffer(ctx, &mut self.info_buffer, 0, bytemuck::cast_slice(data))?;
        Ok(())
    }

    fn clear(&mut self, ctx: &mut GpuCtx) {
        for (_, mut mesh) in self.meshes.drain() {
            mesh.destroy(ctx);
        }
    }

    fn destroy(&mut self, ctx: &mut GpuCtx, dpool: &mut GpuDsl) {
        for (_, mut mesh) in self.meshes.drain() {
            mesh.destroy(ctx);
        }
        self.info_buffer.destroy(ctx);
        dpool.reclaim(self.dset);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct GpuCamera {
    transform: glam::Mat4,
    eye: glam::Vec4,
}

struct GpuLoadedCamera {
    buffer: GpuBuffer,
    dset: vk::DescriptorSet,
}

impl GpuLoadedCamera {
    fn new(ctx: &mut GpuCtx, dpool: &mut GpuDsl) -> anyhow::Result<Self> {
        let buffer = GpuBuffer::new_gpu_local_ro(
            ctx,
            size_of::<GpuCamera>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        let dset = dpool.get_set(ctx)?;
        unsafe {
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(buffer.handle)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(dset)],
                &[],
            );
        }
        Ok(Self { buffer, dset })
    }

    fn udpate(
        &mut self,
        ctx: &mut GpuCtx,
        cb: &mut GpuCommandRecorder,
        cam: &Camera3d,
    ) -> anyhow::Result<()> {
        let transform = cam.get_perspective_proj();
        let gpu_cam = GpuCamera {
            transform,
            eye: glam::Vec4::from((cam.eye, 1.0)),
        };
        cb.write_to_gpu_buffer(ctx, &mut self.buffer, 0, bytemuck::bytes_of(&gpu_cam))?;
        Ok(())
    }

    fn destroy(&mut self, ctx: &mut GpuCtx, dpool: &mut GpuDsl) {
        self.buffer.destroy(ctx);
        dpool.reclaim(self.dset);
    }
}

pub struct RendererVk12 {
    deferred_command_buffer: Option<GpuCommandRecorder>,
    loaded_materials: GpuMaterialMgr,
    loaded_meshes: GpuMeshMgr,
    loaded_camera: GpuLoadedCamera,
    depth_image: GpuImage,
    mesh_pipeline: RenderPipelineVk12,
    smap_pipeline: RenderPipelineVk12,
    swapchain: GpuSwapchain,
    ctx: GpuCtx,
}

impl RendererVk12 {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let mut ctx = GpuCtx::new(window)?;
        let swapchain = GpuSwapchain::new(&mut ctx)?;
        let smap_pipeline = RenderPipelineVk12::new_smap_pipeline(&mut ctx)?;
        let mut mesh_pipeline =
            RenderPipelineVk12::new_mesh_pipeline(&mut ctx, swapchain.format.format)?;
        let loaded_materials = GpuMaterialMgr::new(&mut ctx, &mut mesh_pipeline.dsls[2])?;
        let loaded_meshes = GpuMeshMgr::new(&mut ctx, &mut mesh_pipeline.dsls[1])?;
        let loaded_camera = GpuLoadedCamera::new(&mut ctx, &mut mesh_pipeline.dsls[0])?;
        let depth_image = GpuImage::new(
            &mut ctx,
            vk::ImageType::TYPE_2D,
            vk::Format::D32_SFLOAT,
            (swapchain.res.0, swapchain.res.1, 1),
            1,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        Ok(Self {
            deferred_command_buffer: None,
            loaded_materials,
            loaded_meshes,
            loaded_camera,
            depth_image,
            mesh_pipeline,
            smap_pipeline,
            swapchain,
            ctx,
        })
    }

    fn get_deferred_cmd_buffer(&mut self) -> anyhow::Result<GpuCommandRecorder> {
        let cr = self.deferred_command_buffer.take();
        let cr = match cr {
            Some(t) => t,
            None => {
                let cr = self.ctx.get_command_recorder()?;
                cr
            }
        };
        Ok(cr)
    }

    fn load_material(name: &str) -> anyhow::Result<Material> {
        let mat_bytes = fs::read(name)?;
        let material = ron::de::from_bytes(&mat_bytes)?;
        Ok(material)
    }

    pub fn reset_data(&mut self) {
        self.loaded_meshes.clear(&mut self.ctx);
        self.loaded_materials.clear();
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        self.swapchain.resize(&mut self.ctx)?;
        Ok(())
    }

    fn render_shadow_map(
        &mut self,
        cr: &mut GpuCommandRecorder,
        scene: &Scene,
        light: &Light,
        output: &mut GpuImage,
    ) -> anyhow::Result<()> {
        cr.begin(&mut self.ctx)?;
        self.smap_pipeline
            .start(&mut self.ctx, cr.cb, vec![output])?;
        unsafe {
            for (i, drawable) in scene.drawables.iter().enumerate() {
                let Some(gpu_mesh) = self.loaded_meshes.meshes.get(&drawable.mesh) else {
                    continue;
                };
                let pc_data = MeshPipelinePushConstant {
                    obj_id: i as _,
                    material_id: 0,
                };
                self.ctx
                    .device
                    .cmd_bind_vertex_buffers(cr.cb, 0, &[gpu_mesh.vbo.handle], &[0]);
                self.ctx.device.cmd_bind_index_buffer(
                    cr.cb,
                    gpu_mesh.ibo.handle,
                    0,
                    vk::IndexType::UINT16,
                );
                self.ctx.device.cmd_push_constants(
                    cr.cb,
                    self.mesh_pipeline.layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    bytemuck::bytes_of(&pc_data),
                );
                self.ctx
                    .device
                    .cmd_draw_indexed(cr.cb, gpu_mesh.draw_count, 1, 0, 0, 0);
            }
            self.ctx.device.cmd_end_render_pass(cr.cb);
        }
        todo!()
    }

    pub fn render(&mut self, scene: &Scene, camera: &Camera3d) -> anyhow::Result<()> {
        let Some(idx) = self.swapchain.acquire(&mut self.ctx)? else {
            self.resize()?;
            return Ok(());
        };

        let mut cr = self.get_deferred_cmd_buffer()?;

        let object_datas: Vec<_> = scene
            .drawables
            .iter()
            .map(|d| GpuMeshInfo {
                transform: d.transform,
            })
            .collect();
        self.loaded_meshes
            .write_data(&mut self.ctx, &mut cr, &object_datas)?;
        self.loaded_camera.udpate(&mut self.ctx, &mut cr, camera)?;

        let mut material_ids = vec![];
        for drawable in &scene.drawables {
            match self.loaded_materials.materials.get(&drawable.material) {
                Some(idx) => {
                    material_ids.push(*idx);
                }
                None => {
                    let material = Self::load_material(&drawable.material).unwrap_or_default();
                    self.loaded_materials.add_material(
                        &mut self.ctx,
                        &drawable.material,
                        material,
                        &mut cr,
                    )?;
                    material_ids.push(self.loaded_materials.buffer.len);
                }
            };
            if self.loaded_meshes.meshes.get(&drawable.mesh).is_none() {
                let mesh_bytes = fs::read(&drawable.mesh)?;
                let mesh = ron::de::from_bytes(&mesh_bytes)?;
                self.loaded_meshes
                    .add_mesh(&mut self.ctx, &drawable.mesh, mesh, &mut cr)?;
            }
        }

        if self.depth_image.res != self.swapchain.images[idx as usize].res {
            let depth_image = GpuImage::new(
                &mut self.ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::D32_SFLOAT,
                (self.swapchain.res.0, self.swapchain.res.1, 1),
                1,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )?;
            self.depth_image.destroy(&mut self.ctx);
            self.depth_image = depth_image;
        }
        self.mesh_pipeline.start(
            &mut self.ctx,
            cr.cb,
            vec![
                &mut self.swapchain.images[idx as usize],
                &mut self.depth_image,
            ],
        )?;
        unsafe {
            self.ctx.device.cmd_bind_descriptor_sets(
                cr.cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.mesh_pipeline.layout,
                0,
                &[
                    self.loaded_camera.dset,
                    self.loaded_meshes.dset,
                    self.loaded_materials.dset,
                ],
                &[],
            );

            for (i, drawable) in scene.drawables.iter().enumerate() {
                let Some(gpu_mesh) = self.loaded_meshes.meshes.get(&drawable.mesh) else {
                    continue;
                };
                let pc_data = MeshPipelinePushConstant {
                    obj_id: i as _,
                    material_id: material_ids[i],
                };
                self.ctx
                    .device
                    .cmd_bind_vertex_buffers(cr.cb, 0, &[gpu_mesh.vbo.handle], &[0]);
                self.ctx.device.cmd_bind_index_buffer(
                    cr.cb,
                    gpu_mesh.ibo.handle,
                    0,
                    vk::IndexType::UINT16,
                );
                self.ctx.device.cmd_push_constants(
                    cr.cb,
                    self.mesh_pipeline.layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    bytemuck::bytes_of(&pc_data),
                );
                self.ctx
                    .device
                    .cmd_draw_indexed(cr.cb, gpu_mesh.draw_count, 1, 0, 0, 0);
            }

            self.ctx.device.cmd_end_render_pass(cr.cb);
        }

        self.swapchain.images[idx as usize].transition(
            &mut self.ctx,
            cr.cb,
            GpuImageAccess {
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                access: vk::AccessFlags::empty(),
                stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            },
        );

        let task = cr.submit(&mut self.ctx)?;
        task.wait(&mut self.ctx)?;
        self.deferred_command_buffer = None;
        self.swapchain.present(&mut self.ctx, idx)?;
        Ok(())
    }
}

impl Drop for RendererVk12 {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.ctx.device.device_wait_idle() {
                log::warn!("waiting for gpu to be idle failed: {e}")
            };
            if let Some(mut dcr) = self.deferred_command_buffer.take() {
                dcr.destroy(&mut self.ctx);
            }
            self.loaded_materials
                .destroy(&mut self.ctx, &mut self.mesh_pipeline.dsls[2]);
            self.loaded_meshes
                .destroy(&mut self.ctx, &mut self.mesh_pipeline.dsls[1]);
            self.loaded_camera
                .destroy(&mut self.ctx, &mut self.mesh_pipeline.dsls[0]);
            self.mesh_pipeline.destroy(&mut self.ctx);
            self.depth_image.destroy(&mut self.ctx);
            self.smap_pipeline.destroy(&mut self.ctx);
            self.swapchain.destroy(&mut self.ctx);
        }
    }
}
