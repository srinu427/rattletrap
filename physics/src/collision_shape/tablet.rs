pub struct Tablet {
    pub center: glam::Vec3,
    pub normal: glam::Vec3,
    pub u: glam::Vec3,
    pub v: glam::Vec3,
    pub radius: f32,
}

impl Tablet {
    pub fn plane_equation(&self) -> glam::Vec4 {
        glam::Vec4::new(
            self.normal.x,
            self.normal.y,
            self.normal.z,
            -self.normal.dot(self.center),
        )
    }
}
