use std::sync::Arc;

use glam::Vec4Swizzles;
use hashbrown::HashMap;

use crate::{
    collision_shape::{CollisionShape, ContactState, Orientation},
    utils::remove_component,
};

pub mod collision_shape;
mod utils;

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

fn validate_sep_plane(pl: glam::Vec4, points: &[glam::Vec4]) -> (f32, Vec<usize>) {
    let mut min_dist = 0.0;
    let mut min_dist_points = vec![];
    for (i, p) in points.iter().enumerate() {
        let dist = p.dot(pl);
        if min_dist > dist {
            min_dist = dist;
            min_dist_points = vec![i];
        } else if min_dist == dist {
            min_dist_points.push(i);
        }
    }
    (min_dist, min_dist_points)
}

#[derive(Clone)]
pub struct RigidBody {
    pub mass: f32,
    pub shape: Arc<CollisionShape>,
    pub orient: Orientation,
    pub kinematics: Kinematics,
    pub can_rotate: bool,
    pub has_gravity: bool,
    pub dont_interact_mask: u32,
}

impl RigidBody {
    pub fn make_fut(&self) -> Self {
        let mut out = self.clone();
        out.orient.trans += out.kinematics.velocity + 0.5 * out.kinematics.acceleration;
        out.kinematics.velocity += out.kinematics.acceleration;
        out
    }
}

pub struct PhysicsManager {
    pub objects: Vec<RigidBody>,
    pub object_ids: HashMap<String, usize>,
    pub contacts: Vec<Vec<ContactState>>,
}

impl PhysicsManager {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            object_ids: HashMap::new(),
            contacts: Vec::new(),
        }
    }

    pub fn add_obj(&mut self, name: &str, obj: RigidBody) -> anyhow::Result<()> {
        todo!()
    }

    pub fn forward_ms(&mut self) {
        for obj in &mut self.objects {
            if obj.has_gravity {
                obj.kinematics.acceleration += 10.0 * 1000.0 * 1000.0 * glam::Vec3::NEG_Y;
            }
        }
        let mut touching_objs = vec![vec![]; self.objects.len()];
        for i in 0..self.objects.len() {
            let mut sliding_planes = vec![];
            // Detect sliding surfaces
            for j in (i + 1)..self.objects.len() {
                let contact = &self.contacts[i][j];
                if contact.min_dist == 0.0 {
                    touching_objs[i].push(j);
                    touching_objs[j].push(i);
                    sliding_planes.push(contact.pl);
                }
            }
            for pl in &sliding_planes {
                remove_component(&mut self.objects[i].kinematics.acceleration, pl.xyz());
                remove_component(&mut self.objects[i].kinematics.velocity, pl.xyz());
            }
        }
        todo!()
    }
}
