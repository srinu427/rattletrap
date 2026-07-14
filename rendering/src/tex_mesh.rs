use glam::Vec3Swizzles;

use crate::vkraii::resource::BufferRaii;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub normal: glam::Vec2,
    pub tangent: glam::Vec2,
    pub uv: glam::Vec2,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl Mesh {
    pub fn new_triangle(a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> Self {
        let indices = vec![0, 1, 2];
    }
}

pub struct UploadedMesh {
    pub vertex_buffer: BufferRaii,
    pub index_buffer: BufferRaii,
    pub index_count: u32,
}

pub struct TexMeshPipeline {}
