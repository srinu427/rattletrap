use bytemuck::NoUninit;

#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub obj_id: u32,
}

#[derive(Clone, Debug)]
pub struct Triangle {
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct TriMesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
    pub indices: Vec<u32>,
}

pub fn make_square() -> TriMesh {
    let verts = vec![
        Vertex {
            position: [-0.5, -0.5, 0.0],
            tex_coords: [0.0, 0.0],
            obj_id: 0,
        },
        Vertex {
            position: [0.5, -0.5, 0.0],
            tex_coords: [1.0, 0.0],
            obj_id: 0,
        },
        Vertex {
            position: [0.5, 0.5, 0.0],
            tex_coords: [1.0, 1.0],
            obj_id: 0,
        },
        Vertex {
            position: [-0.5, 0.5, 0.0],
            tex_coords: [0.0, 1.0],
            obj_id: 0,
        },
    ];
    let tris = vec![
        Triangle {
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        Triangle {
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
    ];
    let indices = vec![0, 1, 2, 2, 3, 0];
    TriMesh {
        vertices: verts,
        triangles: tris,
        indices,
    }
}
