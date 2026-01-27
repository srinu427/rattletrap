use bytemuck::NoUninit;

#[derive(Debug, Clone, Copy, NoUninit)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec4,
    pub uv: glam::Vec4,
    pub n: glam::Vec4,
    pub t: glam::Vec4,
    pub bt: glam::Vec4,
}

#[derive(Clone)]
pub struct Mesh {
    pub name: String,
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
        Self {
            name: name.to_string(),
            verts,
            idxs,
        }
    }

    pub fn merge(name: &str, meshes: Vec<Self>) -> Self {
        let mut out = Self {
            name: name.to_string(),
            verts: vec![],
            idxs: vec![],
        };
        for mesh in meshes {
            let Self {
                name: _,
                verts,
                mut idxs,
            } = mesh;
            let voffset = out.verts.len();
            for idx in &mut idxs {
                *idx += voffset as u16;
            }
            out.verts.extend(verts);
            out.idxs.extend(idxs);
        }
        out
    }

    pub fn cube_cuvh(name: &str, c: glam::Vec3, u: glam::Vec3, v: glam::Vec3, hlen: f32) -> Self {
        let h = (hlen / 2.0) * u.cross(v).normalize();
        let faces = vec![
            Self::rect_cuv("f0", c + h, u, v),
            Self::rect_cuv("f1", c - h, -u, v),
            Self::rect_cuv("f2", c + u, v, h),
            Self::rect_cuv("f3", c - u, -v, h),
            Self::rect_cuv("f4", c + v, h, u),
            Self::rect_cuv("f5", c - v, -h, u),
        ];

        Self::merge(name, faces)
    }
}
