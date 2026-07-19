use glam::{Vec3, Vec4Swizzles};

use crate::{orient::Orientation, utils::point_vec4};

#[derive(Debug, Clone)]
pub struct Sphere {
    pub(crate) center: Vec3,
    pub(crate) radius: f32,
}

#[derive(Debug, Clone)]
pub struct Capsule {
    pub(crate) a: Vec3,
    pub(crate) b: Vec3,
    pub(crate) radius: f32,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub(crate) center: Vec3,
    pub(crate) points: Vec<Vec3>,
    pub(crate) edges: Vec<[u32; 2]>,
    pub(crate) faces: Vec<u32>,
}

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere(Sphere),
    Capsule(Capsule),
    Mesh(Mesh),
}

impl CollisionShape {
    pub fn center_hint(&self) -> Vec3 {
        match self {
            Self::Sphere(sphere) => sphere.center,
            Self::Capsule(capsule) => (capsule.a + capsule.b) / 2.0,
            Self::Mesh(mesh) => mesh.center,
        }
    }

    pub fn with_orientation(&self, orientation: &Orientation) -> Self {
        let tr = orientation.to_transform();
        match self {
            Self::Sphere(sphere) => Self::Sphere(Sphere {
                center: (tr * point_vec4(sphere.center)).xyz(),
                radius: sphere.radius,
            }),
            Self::Capsule(capsule) => Self::Capsule(Capsule {
                a: (tr * point_vec4(capsule.a)).xyz(),
                b: (tr * point_vec4(capsule.a)).xyz(),
                radius: capsule.radius,
            }),
            Self::Mesh(mesh) => {
                let mut new_points = mesh.points.clone();
                for p in new_points.iter_mut() {
                    *p = (orientation.rotation * point_vec4(*p)).xyz();
                    *p += orientation.translation;
                }
                let mut new_center = mesh.center;
                new_center = (orientation.rotation * point_vec4(new_center)).xyz();
                new_center += orientation.translation;
                Self::Mesh(Mesh {
                    center: new_center,
                    points: new_points,
                    edges: mesh.edges.clone(),
                    faces: mesh.faces.clone(),
                })
            }
        }
    }

    pub fn farthest_point_along(&self, dir: Vec3) -> Vec3 {
        match self {
            Self::Sphere(sphere) => sphere.center + (dir * sphere.radius),
            Self::Capsule(capsule) => {
                let a_dist = capsule.a + (dir * capsule.radius);
                let b_dist = capsule.b + (dir * capsule.radius);
                a_dist.max(b_dist)
            }
            Self::Mesh(mesh) => {
                let mut max_dist = mesh.points[0].dot(dir);
                let mut max_dist_point = mesh.points[0];
                for &p in &mesh.points[1..] {
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
}
