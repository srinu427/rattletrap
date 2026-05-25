use std::sync::Arc;

use hashbrown::HashMap;

use crate::{
    collision_shape::CollisionShape, intersection_info::IntersectionInfo, orient::Orientation,
};

pub mod collision_shape;
pub mod intersection_info;
pub mod orient;
mod utils;

#[derive(Debug, Clone)]
pub struct Kinematics {
    pub velocity: glam::Vec3,
    pub acceleration: glam::Vec3,
}

impl Kinematics {
    pub fn new() -> Self {
        Self {
            velocity: glam::Vec3::ZERO,
            acceleration: glam::Vec3::ZERO,
        }
    }
}

#[derive(Clone)]
pub struct RigidBody {
    pub mass: f32,
    pub shape: Arc<Box<dyn CollisionShape>>,
    pub orient: Orientation,
    orient_shape: Box<dyn CollisionShape>,
    pub kinematics: Kinematics,
    pub can_rotate: bool,
    pub has_gravity: bool,
    pub dont_interact_mask: u32,
    stuck: bool,
}

impl RigidBody {
    pub fn new(
        mass: f32,
        shape: Arc<Box<dyn CollisionShape>>,
        initial_orient: Orientation,
        initial_kin: Kinematics,
        can_rotate: bool,
        has_gravity: bool,
        dont_interact_mask: u32,
    ) -> Self {
        let orient_shape = shape.with_orientation(&initial_orient);
        Self {
            mass,
            shape,
            orient: initial_orient,
            orient_shape,
            kinematics: initial_kin,
            can_rotate,
            has_gravity,
            dont_interact_mask,
            stuck: false,
        }
    }

    pub fn is_stuck(&self) -> bool {
        self.stuck
    }

    pub fn refresh_orient_shape(&mut self) {
        self.orient_shape = self.shape.with_orientation(&self.orient);
    }

    pub fn make_fut_ms(&self) -> Self {
        let mut out = self.clone();
        out.orient.translation +=
            out.kinematics.velocity * 0.001 + 0.5 * out.kinematics.acceleration * 0.001 * 0.001;
        out.kinematics.velocity += out.kinematics.acceleration * 0.001;
        out.refresh_orient_shape();
        out
    }
}

pub struct PhysicsManager {}

impl PhysicsManager {
    pub fn run(&mut self, rigid_bodies: &mut [RigidBody]) {
        let rb_count = rigid_bodies.len();
        let mut old_inter_map = HashMap::new();
        for i in 0..rb_count {
            for j in i + 1..rb_count {
                let Some(before_inter) = IntersectionInfo::new(
                    &rigid_bodies[i].orient_shape,
                    &rigid_bodies[j].orient_shape,
                ) else {
                    continue;
                };
                old_inter_map.insert((i, j), before_inter);
            }
        }
    }
}
