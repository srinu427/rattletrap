use crate::collision_shape::{CollisionShape, Orientation, rectangle::Rectangle, sphere::Sphere};

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
        let sep_plane = glam::Vec4::from((d_dir, s1.radius));
        return Some(SepPlane {
            rel_to_first: true,
            plane: sep_plane,
        });
    }
    None
}

fn sp_rect_sep(s: &Sphere, r: &Rectangle) -> Option<SepPlane> {
    let pl_c_dist = r.pl.dot(glam::Vec4::from((s.center, 1.0)));
    if pl_c_dist > s.radius {
        return Some(SepPlane {
            rel_to_first: false,
            plane: r.pl,
        });
    }
    for ep in &r.edge_planes {
        let ep_c_dist = ep.dot(glam::Vec4::from((s.center, 1.0)));
        if ep_c_dist > s.radius {
            return Some(SepPlane {
                rel_to_first: false,
                plane: *ep,
            });
        }
    }
    None
}

fn points_only_one_side(pl: glam::Vec4, points: &[glam::Vec4]) -> bool {
    let mut test_is_pos: Option<bool> = None;
    for p in points {
        let f_p_dist = p.dot(pl);
        if f_p_dist != 0.0 {
            let is_pos = f_p_dist > 0.0;
            match test_is_pos {
                Some(tip) => {
                    if tip ^ is_pos {
                        return true;
                    }
                }
                None => {
                    test_is_pos = Some(is_pos);
                }
            };
        };
    }
    false
}

fn rect_rect_sep(r1: &Rectangle, r2: &Rectangle) -> Option<SepPlane> {
    if points_only_one_side(r1.pl, &r2.points) {
        return Some(SepPlane {
            rel_to_first: true,
            plane: r1.pl,
        });
    }
    if points_only_one_side(r2.pl, &r1.points) {
        return Some(SepPlane {
            rel_to_first: false,
            plane: r2.pl,
        });
    }
    None
}
