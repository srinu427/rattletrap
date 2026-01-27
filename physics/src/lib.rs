use crate::collision_shape::{CollisionShape, Orientation};

pub mod collision_shape;

#[derive(Debug, Clone)]
pub struct Kinematics {
    velocity: glam::Vec3,
    acceleration: glam::Vec3,
}

#[derive(Debug, Clone)]
pub struct Dynamics {
    mass: f32,
    net_force: glam::Vec3,
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    name: String,
    shape: CollisionShape,
    orientation: Orientation,
    kinematics: Kinematics,
    dynamics: Dynamics,
}
