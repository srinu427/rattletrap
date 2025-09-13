use crate::collision_shape::{capsule::Capsule, sphere::Sphere, tablet::Tablet};

pub mod sphere;
pub mod capsule;
pub mod tablet;

pub enum CollisionShape {
    Sphere(Sphere),
    Capsule(Capsule),
    Tablet(Tablet),
}

pub fn sphere_sphere_pen(s1: &Sphere, s2: &Sphere) -> Option<(f32, glam::Vec3)> {
    let delta = s2.center - s1.center;
    let dist = delta.length();
    let penetration = s1.radius + s2.radius - dist;
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
    let ab = capsule.point_b - capsule.point_a;
    let t = (sphere.center - capsule.point_a).dot(ab) / ab.dot(ab);
    let t = t.clamp(0.0, 1.0);
    let closest_point = capsule.point_a + t * ab;
    let delta = sphere.center - closest_point;
    let dist = delta.length();
    let penetration = sphere.radius + capsule.radius - dist;
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

pub fn sphere_tablet_pen(sphere: &Sphere, tablet: &Tablet) -> Option<(f32, glam::Vec3)> {
    let dist_to_plane = tablet.normal.dot(sphere.center - tablet.center);
    let plane_proj = sphere.center - dist_to_plane * tablet.normal;

    let d = plane_proj - tablet.center;
    let u_dist = d.dot(tablet.u);
    let v_dist = d.dot(tablet.v);

    let u_dist_clamped = u_dist.clamp(-1.0, 1.0);
    let v_dist_clamped = v_dist.clamp(-1.0, 1.0);

    let closest_point = tablet.center + u_dist_clamped * tablet.u + v_dist_clamped * tablet.v;
    let delta = sphere.center - closest_point;

    let dist = delta.length();
    let penetration = sphere.radius + tablet.radius - dist;
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

pub fn capsule_capsule_pen(c1: &Capsule, c2: &Capsule) -> Option<(f32, glam::Vec3)> {
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
    let penetration = c1.radius + c2.radius - dist;
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
