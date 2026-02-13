use std::sync::Arc;

use glam::Vec4Swizzles;
use hashbrown::HashMap;

use crate::collision_shape::{
    CollisionShape, Orientation, convex_mesh::ConvexMesh, sphere::Sphere,
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

#[derive(Debug, Clone)]
pub enum Separation {
    Face {
        of_first: bool,
        face_idx: usize,
    },
    EdgeCross {
        idx1: usize,
        idx2: usize,
    },
    EdgeTangent {
        of_first: bool,
        edge_idx: usize,
        normal: glam::Vec3,
    },
    PointTangent {
        of_first: bool,
        point_idx: usize,
        normal: glam::Vec3,
    },
}

impl Separation {
    pub fn swap_first_object(&mut self) {
        match self {
            Separation::Face {
                of_first,
                face_idx: _,
            } => {
                *of_first = !*of_first;
            }
            Separation::EdgeCross { idx1, idx2 } => {
                (*idx1, *idx2) = (*idx2, *idx1);
            }
            Separation::EdgeTangent {
                of_first,
                edge_idx: _,
                normal: _,
            } => {
                *of_first = !*of_first;
            }
            Separation::PointTangent {
                of_first,
                point_idx: _,
                normal: _,
            } => {
                *of_first = !*of_first;
            }
        }
    }
}

pub struct ContactState {
    sep: Separation,
    min_d: f32,
}

impl ContactState {
    pub fn new(a: &CollisionShape, b: &CollisionShape) -> Self {
        todo!()
    }
}

fn sp_sp_sep(s1: &Sphere, s2: &Sphere) -> Option<Separation> {
    let d = s2.center - s1.center;
    let r = s1.radius + s2.radius;
    if d.dot(d) > r * r {
        let d_dir = d.normalize();
        return Some(Separation::PointTangent {
            of_first: true,
            point_idx: 0,
            normal: d_dir,
        });
    }
    None
}

fn sp_ppoly_sep(s: &Sphere, pp: &ConvexMesh) -> Option<Separation> {
    let pl_c_dist = pp.pl.dot(glam::Vec4::from((s.center, 1.0)));
    if pl_c_dist > s.radius {
        return Some(Separation::Face {
            of_first: false,
            face_idx: 0,
        });
    }
    for (i, p) in pp.points.iter().enumerate() {
        let d = p.xyz() - s.center;
        let d_norm = d.normalize();
        let pl = glam::Vec4::from((d_norm, -s.center.dot(d_norm) - s.radius));
        if points_on_pos(pl, &pp.points) {
            return Some(Separation::PointTangent {
                of_first: false,
                point_idx: i,
                normal: d_norm,
            });
        }
    }
    for i in 0..pp.points.len() {
        let j = (i + 1) % pp.points.len();
        let a = pp.points[i];
        let b = pp.points[j];
        let ba = b - a;
        let mut d = b.xyz() - s.center;
        d = d - d.dot(ba.xyz());
        let d_norm = d.normalize();
        let pl = glam::Vec4::from((d_norm, -s.center.dot(d_norm) - s.radius));
        if points_on_pos(pl, &pp.points) {
            return Some(Separation::EdgeTangent {
                of_first: false,
                edge_idx: i,
                normal: d_norm,
            });
        }
    }
    None
}

fn points_on_pos(pl: glam::Vec4, points: &[glam::Vec4]) -> bool {
    for p in points {
        if p.dot(pl) < 0.0 {
            return false;
        }
    }
    true
}

fn points_min_dist(pl: glam::Vec4, points: &[glam::Vec4]) -> f32 {
    let mut min_dist = f32::INFINITY;
    for p in points {
        let f_p_dist = p.dot(pl);
        if f_p_dist < min_dist {
            min_dist = f_p_dist;
        }
    }
    min_dist
}

fn points_min_max_dist(pl: glam::Vec4, points: &[glam::Vec4]) -> (f32, f32) {
    let mut min_dist = f32::INFINITY;
    let mut max_dist = f32::NEG_INFINITY;
    for p in points {
        let f_p_dist = p.dot(pl);
        if f_p_dist > max_dist {
            max_dist = f_p_dist;
        }
        if f_p_dist < min_dist {
            min_dist = f_p_dist;
        }
    }
    (min_dist, max_dist)
}

fn ppoly_ppoly_sep(r1: &ConvexMesh, r2: &ConvexMesh) -> ContactState {
    let mut out = ContactState {
        sep: Separation::Face {
            of_first: false,
            face_idx: 0,
        },
        min_d: f32::NEG_INFINITY,
    };

    // Check Planes
    let min_d = points_min_dist(r1.pl, &r2.points);
    if out.min_d < min_d {
        out.min_d = min_d;
        out.sep = Separation::Face {
            of_first: true,
            face_idx: 0,
        };
        if min_d >= 0.0 {
            return out;
        }
    }
    let min_d = points_min_dist(r2.pl, &r1.points);
    if out.min_d < min_d {
        out.min_d = min_d;
        out.sep = Separation::Face {
            of_first: false,
            face_idx: 0,
        };
        if min_d >= 0.0 {
            return out;
        }
    }

    // Edge Crosses
    for &(i1, j1) in &r1.edges {
        let e1 = r1.points[j1] - r1.points[i1];
        for &(i2, j2) in &r2.edges {
            let e2 = r2.points[j2] - r2.points[i2];
            let n = e1.xyz().cross(e2.xyz());
            if n.length_squared() == 0.0 {
                continue;
            }
            let pl = glam::Vec4::from((n, -n.dot(r1.points[i1].xyz())));
            // Check if it separates
            let (min_d1, max_d1) = points_min_max_dist(pl, &r1.points);
            let (min_d2, max_d2) = points_min_max_dist(pl, &r2.points);
            if min_d1 < 0.0 && max_d1 > 0.0 {
                continue;
            }
            if min_d2 < 0.0 && max_d2 > 0.0 {
                continue;
            }
            if max_d1 < 0.0 {}
            let side_1 = if min_d >= 0.0 && max_d >= 0.0 {
                true
            } else if min_d < 0.0 && max_d < 0.0 {
                false
            } else {
                continue;
            };

            if side_1 {}
            let side_2 = if min_d >= 0.0 && max_d >= 0.0 {
                true
            } else if min_d < 0.0 && max_d < 0.0 {
                false
            } else {
                continue;
            };
            let Some(side_1) = points_min_max_dist(pl, &r1.points) else {
                continue;
            };
            let Some(side_2) = points_min_max_dist(pl, &r2.points) else {
                continue;
            };
            if side_1 ^ side_2 {
                return Some(Separation::EdgeCross { idx1: i1, idx2: i2 });
            }
        }
    }
    None
}

fn coll_shape_sep(a: &CollisionShape, b: &CollisionShape) -> Option<Separation> {
    match a {
        CollisionShape::Sphere(s1) => match b {
            CollisionShape::Sphere(s2) => sp_sp_sep(s1, s2),
            CollisionShape::PlanarPolygon(pp2) => sp_ppoly_sep(s1, pp2),
        },
        CollisionShape::PlanarPolygon(pp1) => match b {
            CollisionShape::Sphere(s2) => sp_ppoly_sep(s2, pp1).map(|mut sp| {
                sp.swap_first_object();
                sp
            }),
            CollisionShape::PlanarPolygon(pp2) => ppoly_ppoly_sep(pp1, pp2),
        },
    }
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
    pub separations: Vec<Vec<Separation>>,
}

impl PhysicsManager {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            object_ids: HashMap::new(),
            separations: Vec::new(),
        }
    }

    pub fn add_obj(&mut self, name: &str, obj: RigidBody) -> anyhow::Result<()> {
        todo!()
    }

    pub fn forward_ms(&mut self) {
        let mut touching_objs = vec![vec![]; self.objects.len()];
        for i in 0..self.objects.len() {
            let mut sliding_surface: Vec<glam::Vec4> = vec![];
            // Validate separation planes and detect sliding surfaces
            for j in (i + 1)..self.objects.len() {
                if (self.objects[i].dont_interact_mask & self.objects[j].dont_interact_mask) != 0 {
                    continue;
                }
                let sep_plane = self.separations[i][j - i].clone();
                match sep_plane {
                    Separation::Face { of_first, face_idx } => {
                        let (ref_obj, sec_obj) = if of_first {
                            (&fut_objs[i], &fut_objs[j])
                        } else {
                            (&fut_objs[j], &fut_objs[i])
                        };
                        let shape_orient = ref_obj.shape.with_orientation(&ref_obj.orient);
                        match &shape_orient {
                            CollisionShape::Sphere(_) => unreachable!(),
                            CollisionShape::PlanarPolygon(planar_polygon) => {
                                let plane = planar_polygon.pl;
                                let (min_d, min_d_points) = match shape_orient {
                                    CollisionShape::Sphere(sphere) => validate_sep_plane(
                                        plane,
                                        &[glam::Vec4::from((sphere.center, 1.0))],
                                    ),
                                    CollisionShape::PlanarPolygon(planar_polygon) => {
                                        validate_sep_plane(plane, &planar_polygon.points)
                                    }
                                };
                                if min_d > 0.01 {}
                            }
                        }
                    }
                    Separation::EdgeCross { idx1, idx2 } => todo!(),
                    Separation::EdgeTangent {
                        of_first,
                        edge_idx,
                        normal,
                    } => todo!(),
                    Separation::PointTangent {
                        of_first,
                        point_idx,
                        normal,
                    } => todo!(),
                }
            }
        }
        todo!()
    }
}
