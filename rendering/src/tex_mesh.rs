use std::{collections::HashMap, mem::offset_of, sync::Arc};

use anyhow::Context;
use ash::vk;
use gpu_allocator::MemoryLocation;
use naga::ShaderStage;

use crate::vkraii::{
    command::CommandBufferRaii,
    device::{DeviceDropper, DeviceRaii},
    pipeline::{DescriptorSetLayoutRaii, ShaderRaii},
    resource::BufferRaii,
};

#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec2,
    pub color: glam::Vec3,
}

impl Vertex {
    pub fn attribute_descs() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, pos) as _,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as _,
            },
            // vk::VertexInputAttributeDescription {
            //     location: 2,
            //     binding: 0,
            //     format: vk::Format::R32G32_SFLOAT,
            //     offset: offset_of!(Self, tangent) as _,
            // },
            // vk::VertexInputAttributeDescription {
            //     location: 3,
            //     binding: 0,
            //     format: vk::Format::R32G32_SFLOAT,
            //     offset: offset_of!(Self, uv) as _,
            // },
        ]
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl Mesh {
    pub fn new_triangle(a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> Self {
        let vertices = vec![
            Vertex {
                pos: a,
                color: glam::vec3(1.0, 0.0, 0.0),
            },
            Vertex {
                pos: b,
                color: glam::vec3(0.0, 1.0, 0.0),
            },
            Vertex {
                pos: c,
                color: glam::vec3(0.0, 0.0, 1.0),
            },
        ];
        let indices = vec![0, 1, 2];
        Self { vertices, indices }
    }
}

pub struct GpuMesh {
    pub vertex_buffer: BufferRaii,
    pub index_buffer: BufferRaii,
    pub index_count: u32,
}

impl GpuMesh {
    pub fn new(
        device: &mut DeviceRaii,
        cb: &mut CommandBufferRaii,
        mesh: Mesh,
    ) -> anyhow::Result<GpuMesh> {
        let vb_size = mesh.vertices.len() * size_of::<Vertex>();
        let ib_size = mesh.indices.len() * size_of::<u16>();
        let vertex_buffer = BufferRaii::new(
            &device.device_d,
            &device.allocator,
            &vk::BufferCreateInfo::default()
                .size(vb_size as _)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
            MemoryLocation::GpuOnly,
        )?;
        let index_buffer = BufferRaii::new(
            &device.device_d,
            &device.allocator,
            &vk::BufferCreateInfo::default()
                .size(ib_size as _)
                .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
            MemoryLocation::GpuOnly,
        )?;
        let mut stage_buffer = BufferRaii::new(
            &device.device_d,
            &device.allocator,
            &vk::BufferCreateInfo::default()
                .size((vb_size + ib_size) as _)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            MemoryLocation::CpuToGpu,
        )?;
        let mapped_mem = stage_buffer
            .mem
            .allocation
            .mapped_slice_mut()
            .with_context(|| "cant map stage buffer memory")?;
        mapped_mem[..vb_size].copy_from_slice(bytemuck::cast_slice(&mesh.vertices));
        mapped_mem[vb_size..(vb_size + ib_size)]
            .copy_from_slice(bytemuck::cast_slice(&mesh.indices));
        unsafe {
            device.device_d.device.cmd_copy_buffer(
                cb.command_buffer,
                stage_buffer.buffer,
                vertex_buffer.buffer,
                &[vk::BufferCopy::default().size(vb_size as _)],
            );
            device.device_d.device.cmd_copy_buffer(
                cb.command_buffer,
                stage_buffer.buffer,
                index_buffer.buffer,
                &[vk::BufferCopy::default()
                    .size(ib_size as _)
                    .src_offset(vb_size as _)],
            );
        }
        cb.preserve_buffers.push(stage_buffer);
        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as _,
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FramebufferKey {
    pub views: Vec<vk::ImageView>,
}

pub struct TexMeshPass {
    pub gpu_mesh: GpuMesh,
    pub framebuffers: HashMap<FramebufferKey, vk::Framebuffer>,
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layouts: Vec<DescriptorSetLayoutRaii>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    device_d: Arc<DeviceDropper>,
}

impl TexMeshPass {
    pub fn new(
        device: &mut DeviceRaii,
        sc_format: vk::Format,
        cb: &mut CommandBufferRaii,
    ) -> anyhow::Result<Self> {
        let render_pass = unsafe {
            device.device_d.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[vk::AttachmentDescription::default()
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .format(sc_format)
                        .initial_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .store_op(vk::AttachmentStoreOp::STORE)])
                    .subpasses(&[vk::SubpassDescription::default()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::default()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])])
                    .dependencies(&[
                        vk::SubpassDependency::default()
                            .dependency_flags(vk::DependencyFlags::BY_REGION)
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .dst_subpass(0)
                            .src_access_mask(vk::AccessFlags::MEMORY_READ)
                            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
                            .src_subpass(vk::SUBPASS_EXTERNAL),
                        vk::SubpassDependency::default()
                            .dependency_flags(vk::DependencyFlags::BY_REGION)
                            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                            .dst_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
                            .dst_subpass(vk::SUBPASS_EXTERNAL)
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .src_subpass(0),
                    ]),
                None,
            )?
        };
        let descriptor_set_layouts = vec![];
        let vert_shader = ShaderRaii::load_glsl_str(
            &device.device_d,
            include_str!("shaders/triangle.vert"),
            ShaderStage::Vertex,
        )?;
        let frag_shader = ShaderRaii::load_glsl_str(
            &device.device_d,
            include_str!("shaders/triangle.frag"),
            ShaderStage::Fragment,
        )?;
        let pipeline_layout = unsafe {
            device
                .device_d
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?
        };
        let pipeline = unsafe {
            device
                .device_d
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
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
                        .layout(pipeline_layout)
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .sample_shading_enable(false)
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .cull_mode(vk::CullModeFlags::BACK)
                                .depth_bias_enable(false)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0)
                                .polygon_mode(vk::PolygonMode::FILL),
                        )
                        .render_pass(render_pass)
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(vert_shader.module)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::VERTEX),
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(frag_shader.module)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::FRAGMENT),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(&Vertex::attribute_descs())
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .input_rate(vk::VertexInputRate::VERTEX)
                                        .stride(size_of::<Vertex>() as _),
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
        drop(vert_shader);
        drop(frag_shader);
        let mesh = Mesh::new_triangle(
            glam::vec2(0.0, -0.5),
            glam::vec2(0.5, 0.5),
            glam::vec2(-0.5, 0.5),
        );
        let gpu_mesh = GpuMesh::new(device, cb, mesh)?;
        Ok(Self {
            gpu_mesh,
            framebuffers: HashMap::with_capacity(128),
            render_pass,
            descriptor_set_layouts,
            pipeline_layout,
            pipeline,
            device_d: device.device_d.clone(),
        })
    }

    pub fn get_framebuffer(
        &mut self,
        res: (u32, u32),
        attachments: Vec<vk::ImageView>,
    ) -> anyhow::Result<vk::Framebuffer> {
        let key = FramebufferKey { views: attachments };
        let fb = self.framebuffers.get(&key).cloned();
        let fb = match fb {
            Some(t) => t,
            None => {
                let fb = unsafe {
                    self.device_d.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .attachments(&key.views)
                            .height(res.1)
                            .layers(1)
                            .render_pass(self.render_pass)
                            .width(res.0),
                        None,
                    )?
                };
                if self.framebuffers.len() > 128 {
                    unsafe {
                        for fb in self.framebuffers.values() {
                            self.device_d.device.destroy_framebuffer(*fb, None);
                        }
                    }
                    self.framebuffers.clear();
                }
                self.framebuffers.insert(key, fb);
                fb
            }
        };
        Ok(fb)
    }

    pub fn begin(
        &mut self,
        command_buffer: vk::CommandBuffer,
        res: (u32, u32),
        views: Vec<vk::ImageView>,
    ) -> anyhow::Result<()> {
        let fb = self.get_framebuffer(res, views)?;
        let rect = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: res.0,
                height: res.1,
            },
        };
        unsafe {
            self.device_d.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue::default(),
                    }])
                    .framebuffer(fb)
                    .render_area(rect)
                    .render_pass(self.render_pass),
                vk::SubpassContents::INLINE,
            );
            self.device_d.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: res.0 as _,
                    height: res.1 as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device_d
                .device
                .cmd_set_scissor(command_buffer, 0, &[rect]);
            self.device_d.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        }
        Ok(())
    }

    pub fn end(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device_d.device.cmd_end_render_pass(command_buffer);
        }
    }
}

impl Drop for TexMeshPass {
    fn drop(&mut self) {
        unsafe {
            for fb in self.framebuffers.values() {
                self.device_d.device.destroy_framebuffer(*fb, None);
            }
            self.framebuffers.clear();
            self.device_d.device.destroy_pipeline(self.pipeline, None);
            self.device_d
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor_set_layouts.clear();
            self.device_d
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}
