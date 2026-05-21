use crate::{
    orient::Orientation,
    utils::{new_plane, orient_plane, points_min_dist, points_min_max_dist, points_on_side},
};

pub trait CollisionShape {
    fn center_hint(&self) -> glam::Vec3;
    fn with_orientation(&self, orientation: Orientation) -> Self;
    fn farthest_point_along(&self, dir: glam::Vec3) -> glam::Vec3;
}

pub struct Sphere {
    center: glam::Vec3,
    radius: f32,
}

impl CollisionShape for Sphere {
    fn center_hint(&self) -> glam::Vec3 {
        self.center
    }

    fn with_orientation(&self, orientation: Orientation) -> Self {
        Self {
            center: self.center + orientation.translation,
            radius: self.radius,
        }
    }

    fn farthest_point_along(&self, dir: glam::Vec3) -> glam::Vec3 {
        self.center + (dir * self.radius)
    }
}
