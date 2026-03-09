use std::sync::Arc;

use glam::Vec4Swizzles;
use hashbrown::HashMap;

use crate::{
    collision_shape::{CollisionShape, ContactState, Orientation, Separation},
    utils::{dir_vec4, new_plane, orient_plane, point_vec4, remove_component},
};

pub mod collision_shape;
mod utils;

#[derive(Debug, Clone)]
pub struct Kinematics {
    velocity: glam::Vec3,
    acceleration: glam::Vec3,
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
    pub shape: Arc<CollisionShape>,
    pub orient: Orientation,
    pub orient_shape: CollisionShape,
    pub kinematics: Kinematics,
    pub can_rotate: bool,
    pub has_gravity: bool,
    pub dont_interact_mask: u32,
    pub stuck: bool,
}

impl RigidBody {
    pub fn refresh_orient_shape(&mut self) {
        self.orient_shape = self.shape.with_orientation(&self.orient);
    }

    pub fn make_fut(&self) -> Self {
        let mut out = self.clone();
        out.orient.trans += out.kinematics.velocity + 0.5 * out.kinematics.acceleration;
        out.kinematics.velocity += out.kinematics.acceleration;
        out.refresh_orient_shape();
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

    pub fn clear(&mut self) {
        self.objects.clear();
        self.object_ids.clear();
        self.contacts.clear();
    }

    pub fn add_obj(&mut self, name: &str, obj: RigidBody) {
        self.objects.push(obj);
        let last_obj_id = self.objects.len() - 1;
        self.object_ids.insert(name.to_string(), last_obj_id);
        let new_obj = &self.objects[last_obj_id];
        let contacts: Vec<_> = (0..last_obj_id)
            .map(|i| {
                let o = &self.objects[i];
                ContactState::new(&new_obj.orient_shape, &o.orient_shape)
            })
            .collect();
        self.contacts.push(contacts);
        self.contacts[last_obj_id].push(ContactState::dummy());
        for i in 0..last_obj_id {
            let inv_state = self.contacts[last_obj_id][i].obj_swapped();
            self.contacts[i].push(inv_state);
        }
    }

    pub fn get_obj_transform(&self, name: &str) -> Option<glam::Mat4> {
        let obj_id = self.object_ids.get(name)?;
        Some(self.objects[*obj_id].orient.to_transform())
    }

    pub fn objs_slide(&self, i: usize, j: usize) -> bool {
        let contact = &self.contacts[i][j];
        if contact.min_dist != 0.0 {
            return false;
        }
        match &contact.sep {
            Separation::Face(fid) => {
                let (obj1, obj2) = if contact.of_first {
                    (&self.objects[i], &self.objects[j])
                } else {
                    (&self.objects[j], &self.objects[i])
                };
                let CollisionShape::Mesh(cm) = &obj1.orient_shape else {
                    return false;
                };
                let tp = match &obj2.orient_shape {
                    CollisionShape::Sphere(s2) => {
                        vec![point_vec4(s2.center - (contact.pl.xyz() * s2.radius))]
                    }
                    CollisionShape::Mesh(cm2) => cm2
                        .points
                        .iter()
                        .filter(|p| p.dot(contact.pl) == 0.0)
                        .cloned()
                        .collect(),
                };
                let mut some_point_inside = false;
                for p in tp {
                    let mut point_inside = true;
                    for bp in &cm.face_bounds[*fid] {
                        if bp.dot(p) > 0.0 {
                            point_inside = false;
                            break;
                        }
                    }
                    if point_inside {
                        some_point_inside = true;
                        break;
                    }
                }
                some_point_inside
            }
            Separation::EdgeCross {
                idx1,
                idx2,
                negate: _,
            } => {
                let (obj1, obj2) = if contact.of_first {
                    (&self.objects[i], &self.objects[j])
                } else {
                    (&self.objects[j], &self.objects[i])
                };
                let CollisionShape::Mesh(cm1) = &obj1.orient_shape else {
                    return false;
                };
                let CollisionShape::Mesh(cm2) = &obj2.orient_shape else {
                    return false;
                };
                let a1 = cm1.points[cm1.edges[*idx1].0].xyz();
                let b1 = cm1.points[cm1.edges[*idx1].1].xyz();
                let a2 = cm2.points[cm2.edges[*idx2].0].xyz();
                let b2 = cm2.points[cm2.edges[*idx2].1].xyz();
                let en1 = contact.pl.xyz().cross(b1 - a1);
                let en2 = contact.pl.xyz().cross(b2 - a2);
                let ep1 = new_plane(en1, a1);
                let ep2 = new_plane(en2, a2);
                let a1_d = ep2.dot(point_vec4(a1));
                let b1_d = ep2.dot(point_vec4(b1));
                if (a1_d < 0.0 && b1_d < 0.0) || (a1_d > 0.0 && b1_d > 0.0) {
                    return true;
                }
                let a2_d = ep1.dot(point_vec4(a2));
                let b2_d = ep1.dot(point_vec4(b2));
                if (a2_d < 0.0 && b2_d < 0.0) || (a2_d > 0.0 && b2_d > 0.0) {
                    return true;
                }
                false
            }
            Separation::Relative(_) => true,
        }
    }

    pub fn update_contact_state(
        state: &mut ContactState,
        a: &RigidBody,
        a_old: &RigidBody,
        b: &RigidBody,
        b_old: &RigidBody,
    ) {
        let pl_old = state.pl;
        let (obj1, _obj2) = if state.of_first {
            (a_old, b_old)
        } else {
            (b_old, a_old)
        };
        let (new_obj1, new_obj2) = if state.of_first { (a, b) } else { (b, a) };
        let pl_rel = orient_plane(pl_old, &obj1.orient.reverse().to_transform());
        let pl_new = orient_plane(pl_rel, &new_obj1.orient.to_transform());
        let min_dist_new = new_obj2.orient_shape.plane_min_dist(pl_new);
        state.min_dist = min_dist_new;
        state.pl = pl_new;
        if min_dist_new < 0.0 {
            let new_contact_state = ContactState::new(&a.orient_shape, &b.orient_shape);
            *state = new_contact_state;
        }
    }

    pub fn resolve_penetration_along(
        a: &RigidBody,
        b: &RigidBody,
        dir: glam::Vec3,
    ) -> (Orientation, Orientation) {
        let (min_a, max_a) = a.orient_shape.plane_min_max_dist(dir_vec4(dir));
        let (min_b, max_b) = b.orient_shape.plane_min_max_dist(dir_vec4(dir));
        let pen_depth_a = max_b - min_a;
        let pen_depth_b = max_a - min_b;
        let mut a_orient = a.orient.clone();
        let mut b_orient = b.orient.clone();
        if pen_depth_a < pen_depth_b {
            if a.mass == f32::INFINITY {
                b_orient.trans -= dir * pen_depth_a;
            } else {
                a_orient.trans += dir * pen_depth_a;
            }
        } else {
            if a.mass == f32::INFINITY {
                b_orient.trans += dir * pen_depth_a;
            } else {
                a_orient.trans -= dir * pen_depth_b;
            }
        }
        (a_orient, b_orient)
    }

    pub fn resolve_penetrations(&mut self, obj_id: usize) {
        let mut bound_directions = Vec::with_capacity(3);
        loop {
            let mut pen_id = None;
            for i in 0..self.objects.len() {
                if i != obj_id && self.contacts[obj_id][i].min_dist < 0.0 {
                    pen_id = Some(i);
                    break;
                }
            }
            let Some(pen_id) = pen_id else {
                break;
            };
            let mut pen_dir = self.contacts[obj_id][pen_id].pl.xyz();
            for b in &bound_directions {
                remove_component(&mut pen_dir, *b);
            }
            if pen_dir.length_squared() == 0.0 {
                self.objects[obj_id].stuck = true;
                return;
            }
            let pen_dir = pen_dir.normalize();
            let (obj_new_orient, pen_new_orient) = Self::resolve_penetration_along(
                &self.objects[obj_id],
                &self.objects[pen_id],
                pen_dir,
            );
            let mut new_obj = self.objects[obj_id].clone();
            new_obj.orient = obj_new_orient;
            new_obj.refresh_orient_shape();
            let mut new_pen_obj = self.objects[pen_id].clone();
            new_pen_obj.orient = pen_new_orient;
            new_pen_obj.refresh_orient_shape();
            for i in 0..self.objects.len() {
                if i != obj_id {
                    Self::update_contact_state(
                        &mut self.contacts[obj_id][i],
                        &new_obj,
                        &self.objects[obj_id],
                        &self.objects[i],
                        &self.objects[i],
                    );
                    self.contacts[i][obj_id] = self.contacts[obj_id][i].obj_swapped();
                }
                if i != pen_id {
                    Self::update_contact_state(
                        &mut self.contacts[pen_id][i],
                        &new_pen_obj,
                        &self.objects[pen_id],
                        &self.objects[i],
                        &self.objects[i],
                    );
                    self.contacts[i][pen_id] = self.contacts[pen_id][i].obj_swapped();
                }
            }
            self.objects[obj_id] = new_obj;
            self.objects[pen_id] = new_pen_obj;
            bound_directions.push(pen_dir);
        }
    }

    pub fn forward_ms(&mut self) {
        for obj in &mut self.objects {
            if obj.has_gravity {
                obj.kinematics.acceleration = 0.000001 * glam::Vec3::NEG_Y;
            }
        }
        let obj_count = self.objects.len();
        let mut touches = vec![vec![]; obj_count];
        for i in 0..obj_count {
            let mut touching_planes = vec![];
            // Detect sliding surfaces
            for j in i + 1..obj_count {
                let contact = &self.contacts[i][j];
                if self.objs_slide(i, j) {
                    touches[i].push(j);
                    touches[j].push(i);
                    let pl = if contact.of_first {
                        -contact.pl
                    } else {
                        contact.pl
                    };
                    touching_planes.push(pl);
                }
            }
            for pl in &touching_planes {
                let acc = self.objects[i].kinematics.acceleration;
                let vel = self.objects[i].kinematics.velocity;
                let n = pl.xyz();
                if n.dot(acc) < 0.0 {
                    remove_component(&mut self.objects[i].kinematics.acceleration, pl.xyz());
                }
                if n.dot(vel) < 0.0 {
                    remove_component(&mut self.objects[i].kinematics.velocity, pl.xyz());
                }
            }
        }
        let new_objs: Vec<_> = self.objects.iter().map(|o| o.make_fut()).collect();

        // Update contact states
        for i in 0..obj_count {
            for j in i + 1..obj_count {
                Self::update_contact_state(
                    &mut self.contacts[i][j],
                    &new_objs[i],
                    &self.objects[i],
                    &new_objs[j],
                    &self.objects[j],
                );
                self.contacts[j][i] = self.contacts[i][j].obj_swapped();
            }
        }
        self.objects = new_objs;
        for i in 0..obj_count {
            self.resolve_penetrations(i);
        }

        // print!("{:?}\r", self.contacts[0][1]);
    }
}
