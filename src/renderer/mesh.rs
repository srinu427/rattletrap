use avk12::{
    MemoryLocation,
    ash::vk,
    device::Device,
    resource::{BufferCreateInfo, BufferRef},
};
use bytemuck::NoUninit;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, NoUninit, Serialize, Deserialize)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec4,
    pub uv: glam::Vec4,
    pub n: glam::Vec4,
    pub t: glam::Vec4,
    pub bt: glam::Vec4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshCreateInfo {
    RectCUV {
        c: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
    },
    CubeCUVH {
        c: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        h: f32,
    },
    MeshFile(String),
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub verts: Vec<Vertex>,
    pub idxs: Vec<u16>,
}

impl Mesh {
    fn t_bt(
        pos: &[glam::Vec3],
        uvs: &[glam::Vec4],
        idxs: &[u16; 3],
    ) -> (glam::Vec4, glam::Vec4, glam::Vec4) {
        let edge1 = pos[idxs[1] as usize] - pos[idxs[0] as usize];
        let edge2 = pos[idxs[2] as usize] - pos[idxs[0] as usize];
        let duv1 = uvs[idxs[1] as usize] - uvs[idxs[0] as usize];
        let duv2 = uvs[idxs[2] as usize] - uvs[idxs[0] as usize];
        let n = glam::Vec4::from((edge1.cross(edge2).normalize(), 0.0));
        let f = 1.0f32 / (duv1.x * duv2.y - duv2.x * duv1.y);
        let t = f * glam::vec4(
            duv2.y * edge1.x - duv1.y * edge2.x,
            duv2.y * edge1.y - duv1.y * edge2.y,
            duv2.y * edge1.z - duv1.y * edge2.z,
            0.0,
        );
        let bt = f * glam::vec4(
            -duv2.x * edge1.x - duv1.x * edge2.x,
            -duv2.x * edge1.y - duv1.x * edge2.y,
            -duv2.x * edge1.z - duv1.x * edge2.z,
            0.0,
        );
        (n, t, bt)
    }

    pub fn rect_cuv(c: glam::Vec3, u: glam::Vec3, v: glam::Vec3) -> Self {
        let ulen = u.length();
        let vlen = v.length();
        let vert_pos = [c + u + v, c - u + v, c - u - v, c + u - v];
        let uvs = [
            glam::vec4(ulen, vlen, 0.0, 0.0),
            glam::vec4(-ulen, vlen, 0.0, 0.0),
            glam::vec4(-ulen, -vlen, 0.0, 0.0),
            glam::vec4(ulen, -vlen, 0.0, 0.0),
        ];
        let tris = [[0u16, 1, 2], [2, 3, 0]];
        let verts: Vec<_> = tris
            .iter()
            .map(|idxs| {
                let (n, t, bt) = Mesh::t_bt(&vert_pos, &uvs, idxs);
                idxs.map(|i| Vertex {
                    pos: glam::Vec4::from((vert_pos[i as usize], 1.0)),
                    uv: uvs[i as usize],
                    n,
                    t,
                    bt,
                })
            })
            .flatten()
            .collect();
        let idxs = vec![0, 1, 2, 3, 4, 5];
        Self { verts, idxs }
    }

    pub fn merge(meshes: Vec<Self>) -> Self {
        let mut out = Self {
            verts: vec![],
            idxs: vec![],
        };
        for mesh in meshes {
            let Self { verts, mut idxs } = mesh;
            let voffset = out.verts.len();
            for idx in &mut idxs {
                *idx += voffset as u16;
            }
            out.verts.extend(verts);
            out.idxs.extend(idxs);
        }
        out
    }

    pub fn cube_cuvh(c: glam::Vec3, u: glam::Vec3, v: glam::Vec3, hlen: f32) -> Self {
        let h = (hlen / 2.0) * u.cross(v).normalize();
        let faces = vec![
            Self::rect_cuv(c + h, u, v),
            Self::rect_cuv(c - h, -u, v),
            Self::rect_cuv(c + u, v, h),
            Self::rect_cuv(c - u, -v, h),
            Self::rect_cuv(c + v, h, u),
            Self::rect_cuv(c - v, -h, u),
        ];

        Self::merge(faces)
    }

    pub fn new(info: MeshCreateInfo) -> anyhow::Result<Self> {
        match info {
            MeshCreateInfo::RectCUV { c, u, v } => Ok(Self::rect_cuv(
                glam::Vec3::from_array(c),
                glam::Vec3::from_array(u),
                glam::Vec3::from_array(v),
            )),
            MeshCreateInfo::CubeCUVH { c, u, v, h } => Ok(Self::cube_cuvh(
                glam::Vec3::from_array(c),
                glam::Vec3::from_array(u),
                glam::Vec3::from_array(v),
                h,
            )),
            MeshCreateInfo::MeshFile(_) => todo!(),
        }
    }
}

pub struct GpuMesh {
    pub(crate) vert_buffer: BufferRef,
    pub(crate) indx_buffer: BufferRef,
    pub(crate) indx_count: u32,
}

impl GpuMesh {
    pub fn new(device: &Device, mesh: &Mesh) -> anyhow::Result<Self> {
        let vb_size = (mesh.verts.len() * size_of::<Vertex>()) as u64;
        let ib_size = (mesh.idxs.len() * size_of::<u16>()) as u64;
        let stage_buffer = device.new_buffer(
            BufferCreateInfo::builder()
                .size(vb_size + ib_size)
                .used_for(vk::BufferUsageFlags::TRANSFER_SRC)
                .mem_location(MemoryLocation::CpuToGpu)
                .build(),
        )?;
        stage_buffer.write_cpu(0, bytemuck::cast_slice(&mesh.verts))?;
        stage_buffer.write_cpu(vb_size, bytemuck::cast_slice(&mesh.idxs))?;
        let vert_buffer = device.new_buffer(
            BufferCreateInfo::builder()
                .size(vb_size)
                .used_for(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
                .build(),
        )?;
        let indx_buffer = device.new_buffer(
            BufferCreateInfo::builder()
                .size(ib_size)
                .used_for(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
                .build(),
        )?;
        let mut cr = device.new_task()?;
        cr.copy_b2b(
            stage_buffer.slice(0..vb_size),
            vert_buffer.slice(0..vb_size),
        );
        cr.copy_b2b(
            stage_buffer.slice(vb_size..(vb_size + ib_size)),
            indx_buffer.slice(0..ib_size),
        );
        cr.run()?.wait()?;
        Ok(Self {
            vert_buffer,
            indx_buffer,
            indx_count: mesh.idxs.len() as _,
        })
    }
}
