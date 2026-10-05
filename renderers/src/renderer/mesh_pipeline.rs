use ash::vk;
use bytemuck::NoUninit;
use hashbrown::HashMap;

use crate::{
    MeshVertex,
    helpers::{
        device::{GpuCommandRecorder, GpuCtx},
        image::{GpuDsl, GpuImage, GpuImageViewInfo},
    },
    renderer::{
        DrawableIds, GpuLoadedMesh, camera_descriptor::CameraDescriptor, load_glsl,
        scene_descriptor::SceneDescriptor,
    },
};

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct MeshPipelinePushConstant {
    obj_id: u32,
    material_id: u32,
}

pub struct MeshPipeline {
    render_pass: vk::RenderPass,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    framebuffers: HashMap<Vec<vk::ImageView>, vk::Framebuffer>,
}

impl MeshPipeline {
    pub fn new(
        ctx: &mut GpuCtx,
        format: vk::Format,
        camera_dsl: &mut GpuDsl,
        scene_dsl: &mut GpuDsl,
    ) -> anyhow::Result<Self> {
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
        let layout = unsafe {
            ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[camera_dsl.handle, scene_dsl.handle])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .size(size_of::<MeshPipelinePushConstant>() as _)
                        .stage_flags(vk::ShaderStageFlags::ALL)]),
                None,
            )?
        };
        let vert_shader = load_glsl(
            ctx,
            include_str!("shaders/mesh.vert"),
            naga::ShaderStage::Vertex,
        )?;
        let frag_shader = load_glsl(
            ctx,
            include_str!("shaders/mesh.frag"),
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
            framebuffers: Default::default(),
        })
    }

    fn get_frame_buffer(
        &mut self,
        ctx: &GpuCtx,
        images: &mut [&mut GpuImage; 2],
    ) -> anyhow::Result<vk::Framebuffer> {
        let mut views = vec![];
        for image in images.iter_mut() {
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

    pub fn draw_meshes(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        camera_desc: &CameraDescriptor,
        scene_desc: &SceneDescriptor,
        mesh_buffers: &[GpuLoadedMesh],
        drawables: &[DrawableIds],
        color: &mut GpuImage,
        depth: &mut GpuImage,
    ) -> anyhow::Result<()> {
        let framebuffer = self.get_frame_buffer(ctx, &mut [color, depth])?;
        let res = color.res;
        unsafe {
            ctx.device.cmd_begin_render_pass(
                cr.cb,
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
                .cmd_bind_pipeline(cr.cb, vk::PipelineBindPoint::GRAPHICS, self.handle);
            ctx.device.cmd_set_viewport(
                cr.cb,
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
                cr.cb,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(vk::Extent2D {
                        width: res.0,
                        height: res.1,
                    })],
            );
            ctx.device.cmd_bind_descriptor_sets(
                cr.cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[camera_desc.dset, scene_desc.dset],
                &[],
            );

            for drawable in drawables.iter() {
                let pc_data = MeshPipelinePushConstant {
                    obj_id: drawable.mesh_info_idx,
                    material_id: drawable.material_idx,
                };
                ctx.device.cmd_bind_vertex_buffers(
                    cr.cb,
                    0,
                    &[mesh_buffers[drawable.mesh_buffer_idx as usize].vbo.buffer],
                    &[0],
                );
                ctx.device.cmd_bind_index_buffer(
                    cr.cb,
                    mesh_buffers[drawable.mesh_buffer_idx as usize].ibo.buffer,
                    0,
                    vk::IndexType::UINT16,
                );
                ctx.device.cmd_push_constants(
                    cr.cb,
                    self.layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    bytemuck::bytes_of(&pc_data),
                );
                ctx.device.cmd_draw_indexed(
                    cr.cb,
                    mesh_buffers[drawable.mesh_buffer_idx as usize].draw_count,
                    1,
                    0,
                    0,
                    0,
                );
            }

            ctx.device.cmd_end_render_pass(cr.cb);
        }

        Ok(())
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            for (_, fb) in self.framebuffers.drain() {
                ctx.device.destroy_framebuffer(fb, None);
            }
            ctx.device.destroy_pipeline(self.handle, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
