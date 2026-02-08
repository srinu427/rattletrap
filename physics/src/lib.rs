use glam::Vec4Swizzles;
use hashbrown::HashMap;

use crate::collision_shape::{
    CollisionShape, Orientation, planar_polygon::PlanarPolygon, sphere::Sphere,
};

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

fn sp_ppoly_sep(s: &Sphere, pp: &PlanarPolygon) -> Option<Separation> {
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

fn points_on_which_side(pl: glam::Vec4, points: &[glam::Vec4]) -> Option<bool> {
    let mut test_is_pos: Option<bool> = None;
    for p in points {
        let f_p_dist = p.dot(pl);
        if f_p_dist != 0.0 {
            let is_pos = f_p_dist > 0.0;
            match test_is_pos {
                Some(tip) => {
                    if tip ^ is_pos {
                        return Some(tip);
                    }
                }
                None => {
                    test_is_pos = Some(is_pos);
                }
            };
        };
    }
    if test_is_pos.is_none() {
        Some(true)
    } else {
        None
    }
}

fn ppoly_ppoly_sep(r1: &PlanarPolygon, r2: &PlanarPolygon) -> Option<Separation> {
    // Check Planes
    if points_on_which_side(r1.pl, &r2.points).is_some() {
        return Some(Separation::Face {
            of_first: true,
            face_idx: 0,
        });
    }
    if points_on_which_side(r2.pl, &r1.points).is_some() {
        return Some(Separation::Face {
            of_first: false,
            face_idx: 0,
        });
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
            let Some(side_1) = points_on_which_side(pl, &r1.points) else {
                continue;
            };
            let Some(side_2) = points_on_which_side(pl, &r2.points) else {
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

pub struct RigidBody {
    pub mass: f32,
    pub shape: CollisionShape,
    pub orient: Orientation,
    pub can_rotate: bool,
    pub has_gravity: bool,
    pub dont_interact_mask: u32,
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
        for i in 0..self.objects.len() {
            let mut sliding_surface: Vec<glam::Vec4> = vec![];
            // Validate separation planes and detect sliding surfaces
            for j in (i + 1)..self.objects.len() {
                if (self.objects[i].dont_interact_mask & self.objects[j].dont_interact_mask) != 0 {
                    continue;
                }
            }
        }
        todo!()
    }
}
