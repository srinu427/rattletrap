use crate::collision_shape::{rectangle::Rectangle, sphere::Sphere};

pub mod capsule;
pub mod rectangle;
pub mod sphere;
pub mod triangle;

#[derive(Debug, Clone)]
pub struct Orientation {
    trans: glam::Vec3,
    rot: glam::Mat4,
}

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere(Sphere),
    Rect(Rectangle),
}

impl CollisionShape {
    pub fn with_orientation(&self, orientation: &Orientation) -> Self {
        match self {
            Self::Sphere(sphere) => Self::Sphere(sphere.apply_translation(orientation.trans)),
            Self::Rect(rect) => {
                Self::Rect(rect.with_orientation(orientation.trans, orientation.rot))
            }
        }
    }
}
