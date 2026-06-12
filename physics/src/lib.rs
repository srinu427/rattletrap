use std::sync::Arc;

use glam::Mat4;

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

    pub fn fwd_ms(&mut self) {
        self.orient.translation +=
            self.kinematics.velocity * 0.001 + 0.5 * self.kinematics.acceleration * 0.001 * 0.001;
        self.kinematics.velocity += self.kinematics.acceleration * 0.001;
        self.refresh_orient_shape();
    }

    fn refresh_orient_shape(&mut self) {
        self.orient_shape = self.shape.with_orientation(&self.orient);
    }

    fn apply_orient(&mut self, orientation: &Orientation) {
        self.orient.translation += orientation.translation;
        self.orient.rotation *= orientation.rotation;
        self.refresh_orient_shape();
    }
}

pub struct PhysicsManager {}

impl PhysicsManager {
    pub fn new() -> Self {
        Self {}
    }

    fn resolve_penetrations(&mut self, rigid_bodies: &mut [RigidBody]) {
        let rb_count = rigid_bodies.len();
        // Find penetrations
        let mut touch_dirs = vec![vec![]; rb_count];
        for i in 0..rb_count {
            for j in i + 1..rb_count {
                let Some(touch_info) = IntersectionInfo::new(
                    &rigid_bodies[i].orient_shape,
                    &rigid_bodies[j].orient_shape,
                ) else {
                    continue;
                };
                touch_dirs[i].push((j, touch_info));
            }
        }
        // Resolve Penetrations
        for a in 0..rb_count {
            for (b, inter) in &touch_dirs[a] {
                if inter.dist >= 0.0 {
                    continue;
                }
                let total_mass = rigid_bodies[a].mass + rigid_bodies[*b].mass;
                let a_move_dist = inter.dist * (rigid_bodies[a].mass / total_mass);
                let b_move_dist = -inter.dist * (rigid_bodies[*b].mass / total_mass);
                rigid_bodies[a].apply_orient(&Orientation {
                    translation: a_move_dist * inter.dir,
                    rotation: Mat4::IDENTITY,
                });
                rigid_bodies[*b].apply_orient(&Orientation {
                    translation: b_move_dist * inter.dir,
                    rotation: Mat4::IDENTITY,
                });
            }
        }
    }

    pub fn run_ms(&mut self, rigid_bodies: &mut [RigidBody]) {
        // resolve existing penetrations
        self.resolve_penetrations(rigid_bodies);
        // Find touches
        let rb_count = rigid_bodies.len();
        let mut touch_dirs = vec![vec![]; rb_count];
        for i in 0..rb_count {
            for j in i + 1..rb_count {
                let Some(touch_info) = IntersectionInfo::new(
                    &rigid_bodies[i].orient_shape,
                    &rigid_bodies[j].orient_shape,
                ) else {
                    continue;
                };
                touch_dirs[i].push((j, touch_info.clone()));
                touch_dirs[j].push((i, touch_info.obj_swapped()));
            }
        }
        // Normalize velocities
        for i in 0..rb_count {
            for (j, touch_info) in &touch_dirs[i] {
                let rel_vel =
                    rigid_bodies[i].kinematics.velocity - rigid_bodies[*j].kinematics.velocity;
                let vel_component = rel_vel.dot(touch_info.dir);
                if vel_component < 0.0 {
                    let blocked_vel = rel_vel - vel_component * touch_info.dir;
                    rigid_bodies[i].kinematics.velocity =
                        blocked_vel + rigid_bodies[*j].kinematics.velocity;
                }
            }
        }
        for rb in rigid_bodies.iter_mut() {
            rb.fwd_ms();
        }
        // resolve existing penetrations
        self.resolve_penetrations(rigid_bodies);
    }
}
