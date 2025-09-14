use glam::Vec4Swizzles;

use crate::collision_shape::{capsule::Capsule, sphere::Sphere, triangle::Triangle};

pub mod sphere;
pub mod capsule;
pub mod triangle;

pub enum CollisionShape {
    Sphere(Sphere),
    Capsule(Capsule),
    Triangle(Triangle),
}

pub fn sphere_sphere_pen(s1: &Sphere, s2: &Sphere) -> Option<(f32, glam::Vec3)> {
    let delta = s2.center() - s1.center();
    let dist = delta.length();
    let penetration = s1.radius() + s2.radius() - dist;
    if penetration > 0.0 {
        let normal = if dist > 0.0 {
            delta / dist
        } else {
            glam::Vec3::ZERO
        };
        Some((penetration, normal))
    } else {
        None
    }

}

pub fn sphere_capsule_pen(sphere: &Sphere, capsule: &Capsule) -> Option<(f32, glam::Vec3)> {
    let ab = capsule.point_b() - capsule.point_a();
    let t = (sphere.center() - capsule.point_a()).dot(ab) / ab.dot(ab);
    let t = t.clamp(0.0, 1.0);
    let closest_point = capsule.point_a() + t * ab;
    let delta = sphere.center() - closest_point;
    let dist = delta.length();
    let penetration = sphere.radius() + capsule.radius() - dist;
    if penetration > 0.0 {
        let normal = if dist > 0.0 {
            delta / dist
        } else {
            glam::Vec3::ZERO
        };
        Some((penetration, normal))
    } else {
        None
    }
}

pub fn sphere_triangle_pen(sphere: &Sphere, triangle: &Triangle) -> Option<(f32, glam::Vec3)> {
    let dist_to_plane = triangle.normal().dot(sphere.center() - triangle.points()[0]);
    let plane_proj = sphere.center() - dist_to_plane * triangle.normal();

    let side_dists = triangle
        .bound_planes()
        .map(|bp| bp.dot(glam::Vec4::from((plane_proj, 1.0))));

    let max_side_dist = side_dists.iter().cloned().fold(-f32::INFINITY, f32::max);

    if max_side_dist <= 0.0 {
        let penetration = sphere.radius() + triangle.radius() - dist_to_plane.abs();
        if penetration > 0.0 {
            let normal = if dist_to_plane > 0.0 {
                triangle.normal()
            } else {
                -triangle.normal()
            };
            Some((penetration, normal))
        } else {
            None
        }
    } else {
        let mut min_dist_id = 0;
        let mut min_dist = f32::INFINITY;

        for i in 0..3 {
            if side_dists[i] > 0.0 && side_dists[i] < min_dist {
                min_dist = side_dists[i];
                min_dist_id = i;
            }
        }

        let line_proj = plane_proj + min_dist * triangle.bound_planes()[min_dist_id].xyz();

        let side_p1 = triangle.points()[min_dist_id];
        let side_p2 = triangle.points()[(min_dist_id + 1) % 3];
        let t = (line_proj - side_p1).length() / triangle.side_len()[min_dist_id];
        let t_clamped = t.clamp(0.0, 1.0);

        let min_dist_point = side_p1 + t_clamped * (side_p2 - side_p1);

        let delta = min_dist_point - sphere.center();
        let dist = delta.length();
        let penetration = sphere.radius() + triangle.radius() - dist;
        if penetration > 0.0 {
            let normal = if dist > 0.0 {
                delta / dist
            } else {
                glam::Vec3::ZERO
            };
            Some((penetration, normal))
        } else {
            None
        }
    }
}

pub fn capsule_capsule_pen(c1: &Capsule, c2: &Capsule) -> Option<(f32, glam::Vec3)> {
    let d1 = c1.point_b() - c1.point_a();
    let d2 = c2.point_b() - c2.point_a();

    let normal = d1.cross(d2);
    let normal = normal.normalize();

    let sep_p2 = d2.cross(normal);
    let c1a_sep_dist = sep_p2.dot(c1.point_a());
    let c1b_sep_dist = sep_p2.dot(c1.point_b());

    let t1 = c1a_sep_dist / (c1a_sep_dist - c1b_sep_dist);

    let sep_p1 = d1.cross(normal);
    let c2a_sep_dist = sep_p1.dot(c2.point_a());
    let c2b_sep_dist = sep_p1.dot(c2.point_b());

    let t2 = c2a_sep_dist / (c2a_sep_dist - c2b_sep_dist);

    let t1 = t1.clamp(0.0, 1.0);
    let t2 = t2.clamp(0.0, 1.0);

    let p1 = c1.point_a() + d1 * t1;
    let p2 = c2.point_a() + d2 * t2;

    let delta = p2 - p1;
    let dist = delta.length();
    let penetration = c1.radius() + c2.radius() - dist;
    if penetration > 0.0 {
        let normal = if dist > 0.0 {
            delta / dist
        } else {
            glam::Vec3::ZERO
        };
        Some((penetration, normal))
    } else {
        None
    }
}
