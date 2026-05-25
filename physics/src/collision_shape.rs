use crate::{
    orient::Orientation,
    utils::{new_plane, orient_plane, points_min_dist, points_min_max_dist, points_on_side},
};

pub trait CollisionShape {
    fn center_hint(&self) -> glam::Vec3;
    fn farthest_point_along(&self, dir: glam::Vec3) -> glam::Vec3;
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
    center: glam::Vec3,
    radius: f32,
}

impl CollisionShape for Sphere {
    fn center_hint(&self) -> glam::Vec3 {
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

    fn farthest_point_along(&self, dir: glam::Vec3) -> glam::Vec3 {
        self.center + (dir * self.radius)
    }
}
