#[derive(Debug, Clone, Copy, getset::CopyGetters)]
pub struct Sphere {
    #[getset(get_copy = "pub")]
    center: glam::Vec3,
    #[getset(get_copy = "pub")]
    radius: f32,
}

impl Sphere {
    pub fn new(center: glam::Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    pub fn dist_from_point(&self, point: glam::Vec3) -> f32 {
        (point - self.center).length() - self.radius
    }
}