use crate::collision_shape::{planar_polygon::PlanarPolygon, sphere::Sphere};

pub mod capsule;
pub mod planar_polygon;
pub mod sphere;

#[derive(Debug, Clone)]
pub struct Orientation {
    trans: glam::Vec3,
    rot: glam::Mat4,
}

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere(Sphere),
    PlanarPolygon(PlanarPolygon),
}

impl CollisionShape {
    pub fn with_orientation(&self, orientation: &Orientation) -> Self {
        match self {
            Self::Sphere(sphere) => Self::Sphere(sphere.apply_translation(orientation.trans)),
            Self::PlanarPolygon(rect) => {
                Self::PlanarPolygon(rect.with_orientation(orientation.trans, orientation.rot))
            }
        }
    }
}
