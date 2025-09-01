use bytemuck::NoUninit;

#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Camera {
    pub position: [f32; 4],
    pub look_at: [f32; 4],
    pub view_proj: [[f32; 4]; 4],
}