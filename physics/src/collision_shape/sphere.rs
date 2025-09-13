pub struct Sphere {
    pub center: glam::Vec3,
    pub radius: f32,
}

impl Sphere {
    pub fn dist_from_point(&self, point: glam::Vec3) -> f32 {
        (point - self.center).length() - self.radius
    }
}