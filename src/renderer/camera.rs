use bytemuck::NoUninit;
use getset::{Getters, Setters};

#[derive(Debug, Clone, Copy, NoUninit, Getters, Setters)]
#[repr(C)]
pub struct Cam3d {
    #[getset(get = "pub", set = "pub")]
    eye: glam::Vec3,
    #[getset(get = "pub", set = "pub")]
    fov: f32,
    #[getset(get = "pub", set = "pub")]
    dir: glam::Vec3,
    #[getset(get = "pub", set = "pub")]
    aspect: f32,
    #[getset(get = "pub", set = "pub")]
    up: glam::Vec3,
    padding: u32,
    proj_view: glam::Mat4,
}

impl Cam3d {
    pub fn new(eye: glam::Vec3, dir: glam::Vec3, up: glam::Vec3, fov: f32, aspect: f32) -> Self {
        Self {
            eye,
            fov,
            dir,
            aspect,
            up,
            padding: 0,
            proj_view: glam::Mat4::IDENTITY,
        }
    }

    pub fn update_proj_view(&mut self) {
        let view = glam::Mat4::look_to_rh(self.eye, self.dir, self.up);
        let proj = glam::Mat4::perspective_rh(self.fov, self.aspect, 0.1, 100.0);
        self.proj_view = proj * view;
    }
}
