#[derive(Debug, Clone, Copy, getset::CopyGetters)]
pub struct Sphere {
    pub(crate) center: glam::Vec3,
    pub(crate) radius: f32,
}

impl Sphere {
    pub fn new(center: glam::Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    pub fn dist_from_point(&self, point: glam::Vec3) -> f32 {
        (point - self.center).length() - self.radius
    }

    pub fn apply_translation(&self, trans: glam::Vec3) -> Sphere {
        Sphere {
            center: self.center + trans,
            radius: self.radius,
        }
    }
}
