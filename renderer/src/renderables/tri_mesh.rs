#[repr(C)]
#[derive(Clone, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub obj_id: u32,
}

#[derive(Clone, Debug)]
pub struct Triangle {
    pub indices: [u32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct TriMesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
}
