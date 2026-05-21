#[derive(Debug, Clone)]
pub struct Orientation {
    pub translation: glam::Vec3,
    pub rotation: glam::Mat4,
}

impl Orientation {
    pub fn new() -> Self {
        Self {
            translation: glam::Vec3::ZERO,
            rotation: glam::Mat4::IDENTITY,
        }
    }

    pub fn reverse(&self) -> Self {
        let rev_trans = -self.translation;
        let rev_rot = self.rotation.transpose();
        Self {
            translation: rev_trans,
            rotation: rev_rot,
        }
    }

    pub fn to_transform(&self) -> glam::Mat4 {
        glam::Mat4::from_translation(self.translation) * self.rotation
    }
}
