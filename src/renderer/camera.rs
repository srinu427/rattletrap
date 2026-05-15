use bytemuck::NoUninit;
use getset::{CopyGetters, Setters};
use glam::Vec4Swizzles;

#[derive(Debug, Clone, Copy, NoUninit, CopyGetters, Setters)]
#[repr(C)]
pub struct Cam3d {
    #[getset(get_copy = "pub", set = "pub")]
    eye: glam::Vec3,
    #[getset(get_copy = "pub", set = "pub")]
    fov: f32,
    #[getset(get_copy = "pub", set = "pub")]
    dir: glam::Vec3,
    #[getset(get_copy = "pub", set = "pub")]
    aspect: f32,
    #[getset(get_copy = "pub", set = "pub")]
    up: glam::Vec3,
    padding: u32,
    proj_view: glam::Mat4,
}

impl Cam3d {
    pub fn new(eye: glam::Vec3, dir: glam::Vec3, up: glam::Vec3, fov: f32, aspect: f32) -> Self {
        let mut out = Self {
            eye,
            fov,
            dir,
            aspect,
            up,
            padding: 0,
            proj_view: glam::Mat4::IDENTITY,
        };
        out.update_proj_view();
        out
    }

    pub fn update_proj_view(&mut self) {
        let view = glam::Mat4::look_to_rh(self.eye, self.dir, self.up);
        let proj = glam::Mat4::perspective_rh(self.fov, self.aspect, 0.1, 100.0);
        self.proj_view = proj * view;
    }

    pub fn move_up_down(&mut self, up: glam::Vec3, angle: f32) {
        let rot_axis = up.cross(self.dir);
        if rot_axis.length_squared() == 0.0 {
            return;
        }
        let rot = glam::Mat4::from_axis_angle(rot_axis, angle);
        self.dir = (rot * glam::Vec4::from((self.dir, 0.0))).xyz();
        self.up = (rot * glam::Vec4::from((self.up, 0.0))).xyz();
        self.update_proj_view();
    }

    pub fn move_left_right(&mut self, up: glam::Vec3, angle: f32) {
        let rot = glam::Mat4::from_axis_angle(up, angle);
        self.dir = (rot * glam::Vec4::from((self.dir, 0.0))).xyz();
        self.up = (rot * glam::Vec4::from((self.up, 0.0))).xyz();
        self.update_proj_view();
    }
}
