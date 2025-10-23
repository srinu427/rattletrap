use glam::Vec4Swizzles;

use crate::collision_shape::{capsule::Capsule, sphere::Sphere, triangle::Triangle};

pub mod capsule;
pub mod sphere;
pub mod triangle;

#[derive(Debug, Clone)]
pub struct Orientation {
    trans: glam::Vec3,
    rot: glam::Mat4,
}

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere(Sphere),
    Capsule(Capsule),
    Triangle(Triangle),
}

impl CollisionShape {
    pub fn with_orientation(&self, orientation: &Orientation) -> Self {
        match self {
            Self::Sphere(sphere) => Self::Sphere(sphere.apply_translation(orientation.trans)),
            Self::Capsule(capsule) => {
                Self::Capsule(capsule.apply_orientation(orientation.trans, orientation.rot))
            }
            Self::Triangle(triangle) => {
                Self::Triangle(triangle.apply_orientation(orientation.trans, orientation.rot))
            }
        }
    }

    pub fn min_distance(cs1: &Self, cs2: &Self) -> (f32, glam::Vec3) {
        match (cs1, cs2) {
            (Self::Sphere(s1), Self::Sphere(s2)) => sphere_sphere_dist(s1, s2),
            (Self::Capsule(c1), Self::Capsule(c2)) => capsule_capsule_dist(c1, c2),
            (Self::Sphere(sphere), Self::Capsule(capsule))
            | (Self::Capsule(capsule), Self::Sphere(sphere)) => {
                sphere_capsule_dist(sphere, capsule)
            }
            (Self::Sphere(sphere), Self::Triangle(triangle))
            | (Self::Triangle(triangle), Self::Sphere(sphere)) => {
                sphere_triangle_dist(sphere, triangle)
            }
            (Self::Capsule(capsule), Self::Triangle(triangle))
            | (Self::Triangle(triangle), Self::Capsule(capsule)) => {
                capsule_triangle_dist(capsule, triangle)
            }
            _ => (f32::MAX, glam::Vec3::ZERO),
        }
    }
}

pub struct Seperation {
    plane_a: glam::Vec4,
    plane_b: glam::Vec4,
}

pub fn sphere_sphere_coll_time(
    s1: &Sphere,
    s1_vel: glam::Vec3,
    s2: &Sphere,
    s2_vel: glam::Vec3,
) -> Option<f32> {
    let v = s2_vel - s1_vel;
    let ab = s2.center - s1.center;
    let ab_sq = ab.length_squared();
    let r_sq = s1.radius + s2.radius;
    let r_sq = r_sq * r_sq;
    let v_sq = v.length_squared();
    let v_dot_ab = v.dot(ab);

    let det = (v_dot_ab * v_dot_ab) - (v_sq * (ab_sq - r_sq));
    if det < 0.0 {
        None
    } else if det == 0.0 {
        if v_dot_ab > 0.0 {
            None
        } else {
            Some(-v_dot_ab / v_sq)
        }
    } else {
        let det_sqrt = det.sqrt();
        let mut r1_num = -v_dot_ab - det_sqrt;
        if r1_num < 0.0 {
            r1_num = -v_dot_ab + det_sqrt;
        }
        if r1_num < 0.0 {
            None
        } else {
            Some(r1_num / v_sq)
        }
    }
}

pub fn capsule_sphere_coll_time(
    s: &Sphere,
    s_vel: glam::Vec3,
    c: &Capsule,
    c_vel: glam::Vec3,
) -> f32 {
    0.0
}

pub fn sphere_sphere_dist(s1: &Sphere, s2: &Sphere) -> (f32, glam::Vec3) {
    let delta = s2.center - s1.center;
    let dist = delta.length();
    let min_dist = dist - s1.radius - s2.radius;
    let min_dir = if dist == 0.0 {
        glam::Vec3::ZERO
    } else {
        delta / dist
    };
    (min_dist, min_dir)
}

pub fn sphere_capsule_dist(sphere: &Sphere, capsule: &Capsule) -> (f32, glam::Vec3) {
    let ab = capsule.point_b - capsule.point_a;
    let t = (sphere.center - capsule.point_a).dot(ab) / ab.dot(ab);
    let t = t.clamp(0.0, 1.0);
    let closest_point = capsule.point_a + t * ab;
    let delta = sphere.center - closest_point;
    let dist = delta.length();
    let min_dist = dist - sphere.radius - capsule.radius;
    let min_dir = if dist == 0.0 {
        glam::Vec3::ZERO
    } else {
        delta / dist
    };
    (min_dist, min_dir)
}

pub fn sphere_triangle_dist(sphere: &Sphere, triangle: &Triangle) -> (f32, glam::Vec3) {
    let dist_to_plane = triangle.normal.dot(sphere.center - triangle.points[0]);
    let plane_proj = sphere.center - dist_to_plane * triangle.normal;

    let side_dists = triangle
        .bound_planes
        .map(|bp| bp.dot(glam::Vec4::from((plane_proj, 1.0))));

    let max_side_dist = side_dists.iter().cloned().fold(-f32::INFINITY, f32::max);

    if max_side_dist <= 0.0 {
        let min_dist = dist_to_plane.abs() - sphere.radius - triangle.radius;
        let min_dir = if dist_to_plane > 0.0 {
            triangle.normal
        } else {
            -triangle.normal
        };

        (min_dist, min_dir)
    } else {
        let mut min_dist_id = 0;
        let mut min_dist = f32::INFINITY;

        for i in 0..3 {
            if side_dists[i] > 0.0 && side_dists[i] < min_dist {
                min_dist = side_dists[i];
                min_dist_id = i;
            }
        }

        let line_proj = plane_proj + min_dist * triangle.bound_planes[min_dist_id].xyz();

        let side_p1 = triangle.points[min_dist_id];
        let side_p2 = triangle.points[(min_dist_id + 1) % 3];
        let t = (line_proj - side_p1).length() / triangle.side_len[min_dist_id];
        let t_clamped = t.clamp(0.0, 1.0);

        let min_dist_point = side_p1 + t_clamped * (side_p2 - side_p1);

        let delta = min_dist_point - sphere.center;
        let dist = delta.length();
        let min_dist = dist - sphere.radius - triangle.radius;
        let min_dir = if dist == 0.0 {
            glam::Vec3::ZERO
        } else {
            delta / dist
        };
        (min_dist, min_dir)
    }
}

pub fn capsule_capsule_dist(c1: &Capsule, c2: &Capsule) -> (f32, glam::Vec3) {
    let d1 = c1.point_b - c1.point_a;
    let d2 = c2.point_b - c2.point_a;

    let normal = d1.cross(d2);
    let normal = normal.normalize();

    let sep_p2 = d2.cross(normal);
    let c1a_sep_dist = sep_p2.dot(c1.point_a);
    let c1b_sep_dist = sep_p2.dot(c1.point_b);

    let t1 = c1a_sep_dist / (c1a_sep_dist - c1b_sep_dist);

    let sep_p1 = d1.cross(normal);
    let c2a_sep_dist = sep_p1.dot(c2.point_a);
    let c2b_sep_dist = sep_p1.dot(c2.point_b);

    let t2 = c2a_sep_dist / (c2a_sep_dist - c2b_sep_dist);

    let t1 = t1.clamp(0.0, 1.0);
    let t2 = t2.clamp(0.0, 1.0);

    let p1 = c1.point_a + d1 * t1;
    let p2 = c2.point_a + d2 * t2;

    let delta = p2 - p1;
    let dist = delta.length();
    let min_dist = dist - c1.radius - c2.radius;
    let min_dir = if dist == 0.0 {
        glam::Vec3::ZERO
    } else {
        delta / dist
    };
    (min_dist, min_dir)
}

pub fn capsule_triangle_dist(capsule: &Capsule, triangle: &Triangle) -> (f32, glam::Vec3) {
    // Dist from points to plane
    let pda = triangle.normal.dot(capsule.point_a);
    let pdb = triangle.normal.dot(capsule.point_b);

    let min_pd = pda.min(pdb);

    // Dist from edge to edge
    let min_ed = (0..3)
        .map(|i| {
            let edge_capsule = Capsule::new(
                triangle.points[i],
                triangle.points[i + 1 % 3],
                triangle.radius,
            );
            capsule_capsule_dist(&edge_capsule, capsule)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap_or((f32::MAX, glam::Vec3::ZERO));

    if min_pd < min_ed.0 {
        (min_pd - triangle.radius - capsule.radius, triangle.normal)
    } else {
        min_ed
    }
}

pub fn triangle_triangle_dist(t1: &Triangle, t2: &Triangle) -> (f32, glam::Vec3) {
    // Dist from points to plane
    let min_pd_1 = t2
        .points
        .map(|p| {
            sphere_triangle_dist(
                &Sphere {
                    center: p,
                    radius: t2.radius,
                },
                t1,
            )
        })
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap_or((f32::MAX, glam::Vec3::ZERO));

    let min_pd_2 = t1
        .points
        .map(|p| {
            sphere_triangle_dist(
                &Sphere {
                    center: p,
                    radius: t1.radius,
                },
                t2,
            )
        })
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap_or((f32::MAX, glam::Vec3::ZERO));
    let min_pd = if min_pd_1.0 <= min_pd_2.0 {
        min_pd_1
    } else {
        min_pd_2
    };

    // Dist from edge to edge
    let min_ed = (0..3)
        .map(|i| {
            let c1 = Capsule {
                point_a: t1.points[i],
                point_b: t1.points[(i + 1) % 3],
                radius: t1.radius,
            };
            (0..3)
                .map(|j| {
                    capsule_capsule_dist(
                        &c1,
                        &Capsule {
                            point_a: t2.points[j],
                            point_b: t2.points[(j + 1) % 3],
                            radius: t2.radius,
                        },
                    )
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap_or((f32::MAX, glam::Vec3::ZERO))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap_or((f32::MAX, glam::Vec3::ZERO));

    if min_pd.0 <= min_ed.0 { min_pd } else { min_ed }
}
