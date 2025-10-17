use crate::collision_shape::CollisionShape;

pub mod collision_shape;

#[derive(Debug, Clone)]
pub struct Orientation {
    pos: glam::Vec3,
    rotation: glam::Mat4,
}

#[derive(Debug, Clone)]
pub struct Kinematics {
    velocity: glam::Vec3,
    acceleration: glam::Vec3,
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    name: String,
    shape: CollisionShape,
    orientation: Orientation,
    kinematics: Kinematics,
}

pub fn run_physics_sim(rigid_bodies: &mut[RigidBody]) {
    
}