use std::mem::offset_of;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    // pub tex_coords: [f32; 2],
    // pub normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<ModelVertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: offset_of!(Self, position) as _,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // wgpu::VertexAttribute {
                //     offset: offset_of!(Self, tex_coords) as _,
                //     shader_location: 1,
                //     format: wgpu::VertexFormat::Float32x2,
                // },
                // wgpu::VertexAttribute {
                //     offset: offset_of!(Self, normal) as _,
                //     shader_location: 2,
                //     format: wgpu::VertexFormat::Float32x3,
                // },
            ],
        }
    }
}

pub struct Model {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u16>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            vertices: vec![
                ModelVertex {
                    position: [0.0, 0.5, 0.0],
                },
                ModelVertex {
                    position: [-0.5, -0.5, 0.0],
                },
                ModelVertex {
                    position: [0.5, -0.5, 0.0],
                },
            ],
            indices: vec![0, 1, 2],
        }
    }
}

pub struct ModelGpu {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_len: u32,
}

impl ModelGpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mut model = Model::new();
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (model.vertices.len() * size_of::<ModelVertex>()) as _,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_len = model.indices.len() as _;
        if model.indices.len() % 2 != 0 {
            model.indices.push(0);
        }
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (model.indices.len() * size_of::<u16>()) as _,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&model.vertices));
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&model.indices));
        Self {
            vertex_buffer,
            index_buffer,
            index_len,
        }
    }
}

pub struct MeshPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

impl MeshPipeline {
    pub fn new(device: &wgpu::Device, sc_fmt: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/triangle.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MeshPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(ModelVertex::desc())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: sc_fmt,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline }
    }
}
