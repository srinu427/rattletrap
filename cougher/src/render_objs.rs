use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::vk_wrap::image_2d::Image2d;

#[derive(Clone)]
pub struct GVertex {
    pub pos: glam::Vec4,
    pub normal: glam::Vec4,
    pub uv: glam::Vec2,
    pub tex_id: u32,
    pub padding: u32,
}

pub struct Mesh {
    pub verts: Vec<GVertex>,
    pub inds: Vec<u32>,
}

impl Mesh {
    pub fn rect(c: glam::Vec3, u: glam::Vec3, v: glam::Vec3) -> Self {
        let pos = vec![c + u + v, c - u + v, c - u - v, c + u - v];
        let uvs = vec![
            glam::vec2(1.0, 0.0),
            glam::vec2(0.0, 0.0),
            glam::vec2(0.0, 1.0),
            glam::vec2(1.0, 1.0),
        ];
        let normal = glam::Vec4::from((u.cross(v).normalize(), 1.0));
        let verts = pos
            .into_iter()
            .zip(uvs.into_iter())
            .map(|(p, u)| GVertex {
                pos: glam::Vec4::from((p, 1.0)),
                normal,
                uv: u,
                tex_id: 0,
                padding: 0,
            })
            .collect();
        let inds = vec![0, 1, 2, 2, 3, 0];
        Mesh { verts, inds }
    }
}

pub struct MeshPbrTexture {
    albedo: Image2d,
    normal: Image2d,
    rme: Image2d,
}
