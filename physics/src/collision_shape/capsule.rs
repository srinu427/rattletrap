use glam::Vec4Swizzles;

#[derive(Debug, Clone, Copy, getset::CopyGetters)]
pub struct Capsule {
    pub(crate) point_a: glam::Vec3,
    pub(crate) point_b: glam::Vec3,
    pub(crate) radius: f32,
}

impl Capsule {
    pub fn new(point_a: glam::Vec3, point_b: glam::Vec3, radius: f32) -> Self {
        Self {
            point_a,
            point_b,
            radius,
        }
    }

    pub fn apply_orientation(&self, trans: glam::Vec3, rot: glam::Mat4) -> Self {
        let new_point_a = (rot * glam::Vec4::from((self.point_a, 1.0))).xyz() + trans;
        let new_point_b = (rot * glam::Vec4::from((self.point_b, 1.0))).xyz() + trans;

        Self {
            point_a: new_point_a,
            point_b: new_point_b,
            radius: self.radius,
        }
    }
}
