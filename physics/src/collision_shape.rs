use glam::Vec4Swizzles;

use crate::{
    collision_shape::{convex_mesh::ConvexMesh, sphere::Sphere},
    utils::{new_plane, point_vec4, points_min_dist, points_min_max_dist, points_on_side},
};

pub mod capsule;
pub mod convex_mesh;
pub mod sphere;

#[derive(Debug, Clone)]
pub struct Orientation {
    pub trans: glam::Vec3,
    pub rot: glam::Mat4,
}

impl Orientation {
    pub fn new() -> Self {
        Self {
            trans: glam::Vec3::ZERO,
            rot: glam::Mat4::IDENTITY,
        }
    }

    pub fn reverse(&self) -> Self {
        let rev_trans = -self.trans;
        let rev_rot = self.rot.transpose();
        Self {
            trans: rev_trans,
            rot: rev_rot,
        }
    }

    pub fn to_transform(&self) -> glam::Mat4 {
        glam::Mat4::from_translation(self.trans) * self.rot
    }
}

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere(Sphere),
    Mesh(ConvexMesh),
}

impl CollisionShape {
    pub fn with_orientation(&self, orientation: &Orientation) -> Self {
        match self {
            Self::Sphere(sphere) => Self::Sphere(sphere.apply_translation(orientation.trans)),
            Self::Mesh(rect) => {
                Self::Mesh(rect.with_orientation(orientation.trans, orientation.rot))
            }
        }
    }

    pub fn plane_min_max_dist(&self, pl: glam::Vec4) -> (f32, f32) {
        match self {
            CollisionShape::Sphere(s) => {
                let c_dist = pl.dot(point_vec4(s.center));
                (c_dist - s.radius, c_dist + s.radius)
            }
            CollisionShape::Mesh(cm) => points_min_max_dist(pl, &cm.points),
        }
    }

    pub fn plane_min_dist(&self, pl: glam::Vec4) -> f32 {
        match self {
            CollisionShape::Sphere(s) => pl.dot(point_vec4(s.center)),
            CollisionShape::Mesh(cm) => cm
                .points
                .iter()
                .map(|p| p.dot(pl))
                .min_by(f32::total_cmp)
                .unwrap_or(f32::INFINITY),
        }
    }

    pub fn on_plane_pos(&self, pl: glam::Vec4) -> bool {
        match self {
            Self::Sphere(s) => {
                let pl_d = pl.dot(point_vec4(s.center));
                pl_d >= s.radius
            }
            Self::Mesh(cm) => {
                for p in &cm.points {
                    let pl_d = p.dot(pl);
                    if pl_d < 0.0 {
                        return false;
                    }
                }
                true
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Separation {
    Face(usize),
    EdgeCross {
        idx1: usize,
        idx2: usize,
        negate: bool,
    },
    Relative(glam::Vec4),
}

#[derive(Debug, Clone)]
pub struct ContactState {
    pub sep: Separation,
    pub pl: glam::Vec4,
    pub min_dist: f32,
    pub of_first: bool,
}

impl ContactState {
    pub fn new(a: &CollisionShape, b: &CollisionShape) -> Self {
        match a {
            CollisionShape::Sphere(s1) => match b {
                CollisionShape::Sphere(s2) => sp_sp_contact(s1, s2),
                CollisionShape::Mesh(pp2) => sp_mesh_contact(s1, pp2),
            },
            CollisionShape::Mesh(pp1) => match b {
                CollisionShape::Sphere(s2) => {
                    let mut cs = sp_mesh_contact(s2, pp1);
                    cs.of_first = !cs.of_first;
                    cs
                }
                CollisionShape::Mesh(pp2) => mesh_mesh_contact(pp1, pp2),
            },
        }
    }

    pub fn obj_swapped(&self) -> Self {
        let mut out = self.clone();
        out.of_first = !out.of_first;
        out
    }

    pub fn dummy() -> Self {
        Self {
            sep: Separation::Face(0),
            pl: glam::Vec4::ZERO,
            min_dist: 0.0,
            of_first: true,
        }
    }
}

fn sp_sp_contact(s1: &Sphere, s2: &Sphere) -> ContactState {
    let d = s2.center - s1.center;
    let d_len = d.length();
    let d_norm = d_len.recip() * d;
    let min_dist = d_len - s1.radius - s2.radius;
    let (sep, pl) = if min_dist >= 0.0 {
        let pl = new_plane(d_norm, s1.center + (d * s1.radius));
        (Separation::Relative(pl), pl)
    } else {
        let sep_dist = d_len - s2.radius;
        let pl = new_plane(d_norm, s1.center + (d * sep_dist));
        (Separation::Relative(pl), pl)
    };
    ContactState {
        sep,
        pl,
        of_first: true,
        min_dist,
    }
}

fn mesh_mesh_contact(m1: &ConvexMesh, m2: &ConvexMesh) -> ContactState {
    let mut max_min_dist = f32::NEG_INFINITY;
    let mut max_sep = Separation::Face(0);
    let mut max_of_first = true;
    let mut max_pl = glam::Vec4::ZERO;
    for (fi, face) in m1.faces.iter().enumerate() {
        let min_dist = points_min_dist(*face, &m2.points);
        let sep = Separation::Face(fi);
        if min_dist >= 0.0 {
            return ContactState {
                sep,
                pl: *face,
                min_dist,
                of_first: true,
            };
        }
        if min_dist > max_min_dist {
            max_min_dist = min_dist;
            max_sep = sep;
            max_pl = *face;
            max_of_first = true;
        }
    }
    for (fi, face) in m2.faces.iter().enumerate() {
        let min_dist = points_min_dist(*face, &m1.points);
        let sep = Separation::Face(fi);
        if min_dist >= 0.0 {
            return ContactState {
                sep,
                pl: *face,
                min_dist,
                of_first: false,
            };
        }
        if min_dist > max_min_dist {
            max_min_dist = min_dist;
            max_sep = sep;
            max_pl = *face;
            max_of_first = false;
        }
    }
    for (eid1, edge1) in m1.edges.iter().enumerate() {
        for (eid2, edge2) in m2.edges.iter().enumerate() {
            let a1 = m1.points[edge1.0];
            let b1 = m1.points[edge1.1];
            let a2 = m2.points[edge2.0];
            let b2 = m2.points[edge2.1];
            let e1 = b1 - a1;
            let e2 = b2 - a2;
            let n = e1.xyz().cross(e2.xyz());
            if n.length_squared() == 0.0 {
                continue;
            }
            let n = n.normalize();
            let pl = new_plane(n, a1.xyz());
            let side1 = points_on_side(pl, &m1.points);
            let Some(side1) = side1 else {
                continue;
            };
            let negate = !side1;
            let pl = if negate { -pl } else { pl };
            let min_dist = points_min_dist(pl, &m2.points);
            let sep = Separation::EdgeCross {
                idx1: eid1,
                idx2: eid2,
                negate,
            };
            if min_dist >= 0.0 {
                return ContactState {
                    sep,
                    pl,
                    min_dist,
                    of_first: true,
                };
            }
            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                max_sep = sep;
                max_pl = pl;
                max_of_first = true;
            }
        }
    }
    ContactState {
        sep: max_sep,
        pl: max_pl,
        min_dist: max_min_dist,
        of_first: max_of_first,
    }
}

fn sp_mesh_contact(s: &Sphere, m: &ConvexMesh) -> ContactState {
    let mut max_min_dist = f32::NEG_INFINITY;
    let mut max_sep = Separation::Face(0);
    let mut max_pl = glam::Vec4::ZERO;
    let mut max_of_first = true;
    for (fi, face) in m.faces.iter().enumerate() {
        let center_dist = face.dot(point_vec4(s.center));
        let min_dist = center_dist - s.radius;
        let sep = Separation::Face(fi);
        if min_dist >= 0.0 {
            return ContactState {
                sep,
                pl: *face,
                min_dist,
                of_first: false,
            };
        }
        if min_dist > max_min_dist {
            max_min_dist = min_dist;
            max_sep = sep;
            max_pl = *face;
            max_of_first = false;
        }
    }
    for point in &m.points {
        let point_d = point.xyz() - s.center;
        let point_dist = point_d.length();
        let min_dist = point_dist - s.radius;
        let n = point_dist.recip() * point_d;
        let pl = new_plane(n, point.xyz());
        let sep = Separation::Relative(pl);
        if min_dist >= 0.0 {
            return ContactState {
                sep,
                pl,
                min_dist,
                of_first: false,
            };
        }
        if min_dist > max_min_dist {
            max_min_dist = min_dist;
            max_sep = sep;
            max_pl = pl;
            max_of_first = false;
        }
    }
    for edge in &m.edges {
        let a = m.points[edge.0];
        let b = m.points[edge.1];
        let ac = s.center - a.xyz();
        let ab = (b - a).xyz();
        let perp_point = a.xyz() + (ac.dot(ab) * ab);
        let edge_d = perp_point - s.center;
        let edge_dist = edge_d.length();
        let min_dist = edge_dist - s.radius;
        let n = edge_dist.recip() * edge_d;
        let pl = new_plane(n, perp_point);
        let sep = Separation::Relative(pl);
        if min_dist >= 0.0 {
            return ContactState {
                sep,
                pl,
                min_dist,
                of_first: false,
            };
        }
        if min_dist > max_min_dist {
            max_min_dist = min_dist;
            max_sep = sep;
            max_pl = pl;
            max_of_first = false;
        }
    }
    ContactState {
        sep: max_sep,
        pl: max_pl,
        min_dist: max_min_dist,
        of_first: max_of_first,
    }
}
