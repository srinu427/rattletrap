use glam::{Vec3, Vec4Swizzles};

use crate::{orient::Orientation, utils::point_vec4};

pub trait CollisionShape {
    fn center_hint(&self) -> Vec3;
    fn farthest_point_along(&self, dir: Vec3) -> Vec3;
    fn clone_boxed(&self) -> Box<dyn CollisionShape>;
    fn with_orientation(&self, orientation: &Orientation) -> Box<dyn CollisionShape>;
}

impl Clone for Box<dyn CollisionShape> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

#[derive(Debug, Clone)]
pub struct Sphere {
    center: Vec3,
    radius: f32,
}

impl CollisionShape for Sphere {
    fn center_hint(&self) -> Vec3 {
        self.center
    }

    fn clone_boxed(&self) -> Box<dyn CollisionShape> {
        Box::new(self.clone())
    }

    fn with_orientation(&self, orientation: &Orientation) -> Box<dyn CollisionShape> {
        Box::new(Self {
            center: self.center + orientation.translation,
            radius: self.radius,
        })
    }

    fn farthest_point_along(&self, dir: Vec3) -> Vec3 {
        self.center + (dir * self.radius)
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    points: Vec<Vec3>,
    center: Vec3,
}

impl CollisionShape for Mesh {
    fn center_hint(&self) -> Vec3 {
        self.center
    }

    fn farthest_point_along(&self, dir: Vec3) -> Vec3 {
        let mut max_dist = self.points[0].dot(dir);
        let mut max_dist_point = self.points[0];
        for &p in &self.points[1..] {
            let dist = p.dot(dir);
            if dist > max_dist {
                max_dist = dist;
                max_dist_point = p;
            }
        }
        max_dist_point
    }

    fn clone_boxed(&self) -> Box<dyn CollisionShape> {
        Box::new(self.clone())
    }

    fn with_orientation(&self, orientation: &Orientation) -> Box<dyn CollisionShape> {
        let mut out = self.clone();
        for p in out.points.iter_mut() {
            *p = (orientation.rotation * point_vec4(*p)).xyz();
            *p += orientation.translation;
        }
        out.center = (orientation.rotation * point_vec4(out.center)).xyz();
        out.center += orientation.translation;
        Box::new(out)
    }
}
