use glam::Vec4Swizzles;

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
pub struct RigidBody {
    name: String,
    shape: CollisionShape,
    orientation: Orientation,
    kinematics: Kinematics,
    dynamics: Dynamics,
}

#[derive(Debug, Clone)]
struct SepPlane {
    rel_to_first: bool,
    plane: glam::Vec4,
}

fn sp_sp_sep(s1: &Sphere, s2: &Sphere) -> Option<SepPlane> {
    let d = s2.center - s1.center;
    let r = s1.radius + s2.radius;
    if d.dot(d) > r * r {
        let d_dir = d.normalize();
        let sep_plane = glam::Vec4::from((d_dir, -s1.radius));
        return Some(SepPlane {
            rel_to_first: true,
            plane: sep_plane,
        });
    }
    None
}

fn sp_ppoly_sep(s: &Sphere, pp: &PlanarPolygon) -> Option<SepPlane> {
    let pl_c_dist = pp.pl.dot(glam::Vec4::from((s.center, 1.0)));
    if pl_c_dist > s.radius {
        return Some(SepPlane {
            rel_to_first: false,
            plane: pp.pl,
        });
    }
    for p in &pp.points {
        let d = p.xyz() - s.center;
        let d_norm = d.normalize();
        let pl = glam::Vec4::from((d_norm, -s.center.dot(d_norm) - s.radius));
        if points_on_pos(pl, &pp.points) {
            return Some(SepPlane {
                rel_to_first: false,
                plane: pl,
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
            return Some(SepPlane {
                rel_to_first: false,
                plane: pl,
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
    None
}

fn ppoly_ppoly_sep(r1: &PlanarPolygon, r2: &PlanarPolygon) -> Option<SepPlane> {
    // Check Planes
    if points_on_which_side(r1.pl, &r2.points).is_some() {
        return Some(SepPlane {
            rel_to_first: true,
            plane: r1.pl,
        });
    }
    if points_on_which_side(r2.pl, &r1.points).is_some() {
        return Some(SepPlane {
            rel_to_first: false,
            plane: r2.pl,
        });
    }
    // Edge Crosses
    for i1 in 0..r1.points.len() {
        let j1 = (i1 + 1) % r1.points.len();
        let e1 = r1.points[j1] - r1.points[i1];
        for i2 in 0..r2.points.len() {
            let j2 = (i2 + 1) % r2.points.len();
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
                let n_norm = n.normalize();
                return Some(SepPlane {
                    rel_to_first: true,
                    plane: glam::Vec4::from((n_norm, -n_norm.dot(r1.points[i1].xyz()))),
                });
            }
        }
    }
    None
}
