use bytemuck::NoUninit;

#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Vertex {
    pub position: [f32; 4],
    pub tex_coords: [f32; 2],
    pub obj_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Triangle {
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
    pub bitangent: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct TriMesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
    pub indices: Vec<u32>,
}

impl TriMesh {
    pub fn write_obj_id(&mut self, obj_id: u32) {
        for v in self.vertices.iter_mut() {
            v.obj_id = obj_id;
        }
    }
}

pub fn make_square() -> TriMesh {
    let verts = vec![
        Vertex {
            position: [-0.5, -0.5, 0.0, 1.0],
            tex_coords: [0.0, 0.0],
            obj_id: 0,
            padding: 0,
        },
        Vertex {
            position: [0.5, -0.5, 0.0, 1.0],
            tex_coords: [1.0, 0.0],
            obj_id: 0,
            padding: 0,
        },
        Vertex {
            position: [0.5, 0.5, 0.0, 1.0],
            tex_coords: [1.0, 1.0],
            obj_id: 0,
            padding: 0,
        },
        Vertex {
            position: [-0.5, 0.5, 0.0, 1.0],
            tex_coords: [0.0, 1.0],
            obj_id: 0,
            padding: 0,
        },
    ];
    let tris = vec![
        Triangle {
            normal: [0.0, 0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0, 0.0],
        },
        Triangle {
            normal: [0.0, 0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0, 0.0],
        },
    ];
    let indices = vec![0, 1, 2, 2, 3, 0];
    TriMesh {
        vertices: verts,
        triangles: tris,
        indices,
    }
}
