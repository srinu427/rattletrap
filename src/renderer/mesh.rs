use bytemuck::NoUninit;

#[derive(Debug, Clone, Copy, NoUninit)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec4,
    pub uv: glam::Vec4,
}
