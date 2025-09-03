use bytemuck::NoUninit;
use glam::Vec4Swizzles;

#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Camera {
    pub position: glam::Vec4,
    pub look_at: glam::Vec4,
    pub view_proj: glam::Mat4,
}

impl Camera {
    pub fn new(position: glam::Vec4, look_at: glam::Vec4, fov: f32) -> Self {
        let mut cam = Self {
            position,
            look_at,
            view_proj: glam::Mat4::IDENTITY,
        };
        cam.refresh_vp_matrix(fov, 1.0);
        cam
    }

    pub fn refresh_vp_matrix(&mut self, fov: f32, aspect_ratio: f32) {
        self.view_proj = glam::Mat4::perspective_rh(fov, aspect_ratio, 1.0, 1000.0)
            * glam::Mat4::look_at_rh(
                self.position.xyz(),
                self.position.xyz() + self.look_at.xyz(),
                glam::Vec3 {
                    x: 0.0f32,
                    y: 1.0f32,
                    z: 0.0f32,
                },
            );
    }
}
