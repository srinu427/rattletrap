use bytemuck::NoUninit;

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct GVertex {
    pub pos: [f32; 4],
    pub uv: [f32; 2],
    pub obj_id: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GTriangle {
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
    pub bitangent: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct TriMesh {
    pub vertices: Vec<GVertex>,
    pub triangles: Vec<GTriangle>,
    pub indices: Vec<u32>,
}
