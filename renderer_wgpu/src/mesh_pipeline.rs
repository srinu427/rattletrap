use std::mem::offset_of;

use common::Entity;
use indexmap::IndexMap;

use crate::mesh::{Mesh, Vertex};

pub struct MeshGpu {
    pub vb: wgpu::Buffer,
    pub ib: wgpu::Buffer,
    pub indx_count: u32,
}

pub struct MeshRenderData {
    pub meshes: IndexMap<Entity, MeshGpu>,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    pub layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl MeshRenderData {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let shader_mod = device.create_shader_module(wgpu::include_wgsl!("shaders/mesh.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader_mod,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: offset_of!(Vertex, pos) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: offset_of!(Vertex, uv) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: offset_of!(Vertex, n) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: offset_of!(Vertex, t) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: offset_of!(Vertex, bt) as u64,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_mod,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::all(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Ok(Self {
            meshes: IndexMap::new(),
            bind_group_layouts: vec![],
            layout,
            pipeline,
        })
    }
}
