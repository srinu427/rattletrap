use bytemuck::NoUninit;

#[derive(Debug, Clone, Copy, NoUninit)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec4,
    pub uv: glam::Vec4,
}

#[derive(Clone)]
pub struct Mesh {
    pub name: String,
    pub verts: Vec<Vertex>,
    pub idxs: Vec<u16>,
}

impl Mesh {
    pub fn rect_cuv(name: &str, c: glam::Vec3, u: glam::Vec3, v: glam::Vec3) -> Self {
        let ulen = u.length();
        let vlen = v.length();
        let vert_pos = [c + u + v, c - u + v, c - u - v, c + u - v];
        let uvs = [
            glam::vec4(ulen, vlen, 0.0, 0.0),
            glam::vec4(-ulen, vlen, 0.0, 0.0),
            glam::vec4(-ulen, -vlen, 0.0, 0.0),
            glam::vec4(ulen, -vlen, 0.0, 0.0),
        ];
        let verts = (0..4)
            .map(|i| Vertex {
                pos: glam::Vec4::from((vert_pos[i], 1.0)),
                uv: uvs[i],
            })
            .collect();
        let idxs = vec![0, 1, 2, 2, 3, 0];
        Self {
            name: name.to_string(),
            verts,
            idxs,
        }
    }
}
