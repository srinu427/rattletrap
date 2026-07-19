use glam::Vec4Swizzles;

pub struct Camera {
    pub eye: glam::Vec3,
    pub dir: glam::Vec3,
    pub up: glam::Vec3,
    pub fov: f32,
    pub aspect: f32,
}

impl Camera {
    pub fn to_gpu_data_perspective(&self) -> CameraGpu {
        let view = glam::Mat4::look_to_rh(self.eye, self.dir, self.up);
        let proj = glam::Mat4::perspective_rh(self.fov, self.aspect, 0.1, 100.0);
        CameraGpu {
            transform: proj * view,
        }
    }

    pub fn move_left_right(&mut self, up: glam::Vec3, angle: f32) {
        let rot = glam::Mat4::from_axis_angle(up, angle);
        self.dir = (rot * glam::Vec4::from((self.dir, 0.0))).xyz();
        self.up = (rot * glam::Vec4::from((self.up, 0.0))).xyz();
    }

    pub fn move_up_down(&mut self, up: glam::Vec3, angle: f32) {
        let rot_axis = up.cross(self.dir);
        if rot_axis.length_squared() == 0.0 {
            return;
        }
        let rot_axis = rot_axis.normalize();
        let rot = glam::Mat4::from_axis_angle(rot_axis, angle);
        self.dir = (rot * glam::Vec4::from((self.dir, 0.0))).xyz();
        self.up = (rot * glam::Vec4::from((self.up, 0.0))).xyz();
    }
}

#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub struct CameraGpu {
    pub transform: glam::Mat4,
}
