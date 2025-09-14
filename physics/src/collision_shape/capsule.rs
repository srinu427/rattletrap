#[derive(Debug, Clone, Copy, getset::CopyGetters)]
pub struct Capsule {
    #[getset(get_copy = "pub")]
    point_a: glam::Vec3,
    #[getset(get_copy = "pub")]
    point_b: glam::Vec3,
    #[getset(get_copy = "pub")]
    radius: f32,
}

impl Capsule {
    pub fn new(point_a: glam::Vec3, point_b: glam::Vec3, radius: f32) -> Self {
        Self {
            point_a,
            point_b,
            radius,
        }
    }
}