use bytemuck::NoUninit;

#[derive(Debug, Clone, Copy, NoUninit)]
#[repr(C)]
pub struct Cam3d {
    pub eye: glam::Vec3,
    pub fov: f32,
    pub dir: glam::Vec3,
    pub aspect: f32,
    pub up: glam::Vec3,
    pub padding: u32,
    pub proj_view: glam::Mat4,
}

impl Cam3d {
    pub fn update_proj_view(&mut self) {
        let view = glam::Mat4::look_to_rh(self.eye, self.dir, self.up);
        let proj = glam::Mat4::perspective_rh(self.fov, self.aspect, 0.1, 100.0);
        self.proj_view = proj * view;
    }
}
