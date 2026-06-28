use glam::{Vec3, Vec4Swizzles};

use crate::{orient::Orientation, utils::point_vec4};

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere { center: Vec3, radius: f32 },
    Mesh { points: Vec<Vec3>, center: Vec3 },
}

impl CollisionShape {
    pub fn center_hint(&self) -> Vec3 {
        match self {
            Self::Sphere { center, .. } => *center,
            Self::Mesh { center, .. } => *center,
        }
    }

    pub fn with_orientation(&self, orientation: &Orientation) -> Self {
        match self {
            Self::Sphere { center, radius } => Self::Sphere {
                center: *center + orientation.translation,
                radius: *radius,
            },
            Self::Mesh { points, center } => {
                let mut new_points = points.clone();
                for p in new_points.iter_mut() {
                    *p = (orientation.rotation * point_vec4(*p)).xyz();
                    *p += orientation.translation;
                }
                let mut new_center = *center;
                new_center = (orientation.rotation * point_vec4(new_center)).xyz();
                new_center += orientation.translation;
                Self::Mesh {
                    points: new_points,
                    center: new_center,
                }
            }
        }
    }

    pub fn farthest_point_along(&self, dir: Vec3) -> Vec3 {
        match self {
            Self::Sphere { center, radius } => center + (dir * radius),
            Self::Mesh { points, .. } => {
                let mut max_dist = points[0].dot(dir);
                let mut max_dist_point = points[0];
                for &p in &points[1..] {
                    let dist = p.dot(dir);
                    if dist > max_dist {
                        max_dist = dist;
                        max_dist_point = p;
                    }
                }
                max_dist_point
            }
        }
    }

    pub fn new_rect(c: Vec3, u: Vec3, v: Vec3) -> Self {
        let points = vec![c + u + v, c - u + v, c - u - v, c + u - v];
        Self::Mesh { points, center: c }
    }

    pub fn new_cube(c: Vec3, u: Vec3, v: Vec3, hlen: f32) -> Self {
        let h = u.cross(v).normalize() * hlen;
        let points = vec![
            c + u + v + h,
            c - u + v + h,
            c - u - v + h,
            c + u - v + h,
            c + u + v - h,
            c - u + v - h,
            c - u - v - h,
            c + u - v - h,
        ];
        Self::Mesh { points, center: c }
    }
}
