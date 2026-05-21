use glam::Vec4Swizzles;

use crate::{collision_shape::CollisionShape, utils::get_triangle_plane};

#[derive(Debug, Clone)]
pub struct ClosestInfo {
    pub dir: glam::Vec3,
    pub point_1: glam::Vec3,
    pub point_2: glam::Vec3,
    pub dist: f32,
}

impl ClosestInfo {
    pub fn new<O1: CollisionShape, O2: CollisionShape>(a: &O1, b: &O2) -> Self {
        let mut supp_points = vec![];
        loop {
            let check_dir = if supp_points.len() == 0 {
                (b.center_hint() - a.center_hint()).normalize()
            } else if supp_points.len() == 1 {
                todo!()
            } else if supp_points.len() == 2 {
                todo!()
            } else if supp_points.len() == 3 {
                todo!()
            } else {
                break;
            };
            let farth_point_a = a.farthest_point_along(check_dir);
            let farth_point_b = b.farthest_point_along(-check_dir);
            let supp_point = farth_point_a - farth_point_b;
            supp_points.push(supp_point);
            // Check if new point crossed origin
            if check_dir.dot(supp_point) < 0.0 {
                return Some(true);
            }
            if supp_point == glam::Vec3::ZERO {
                return Self {
                    dir: check_dir,
                    point_1: farth_point_a,
                    point_2: farth_point_b,
                    dist: 0.0,
                };
            }
        }
        todo!()
    }

    pub fn obj_swapped(mut self) -> Self {
        self.dir = -self.dir;
        self
    }
}

fn run_gjk<O1: CollisionShape, O2: CollisionShape>(
    a: &O1,
    b: &O2,
    supp_points: &mut Vec<glam::Vec3>,
    epsilon: f32,
) -> Option<bool> {
    let check_dir = if supp_points.len() == 0 {
        (b.center_hint() - a.center_hint()).normalize()
    } else if supp_points.len() == 1 {
        todo!()
    } else if supp_points.len() == 2 {
        todo!()
    } else if supp_points.len() == 3 {
        todo!()
    } else {
        return None;
    };
    let farth_point_a = a.farthest_point_along(check_dir);
    let farth_point_b = b.farthest_point_along(-check_dir);
    let supp_point = farth_point_a - farth_point_b;
    supp_points.push(supp_point);
    // Check if new point crossed origin
    if check_dir.dot(supp_point) < 0.0 {
        return Some(true);
    }
    if supp_point == glam::Vec3::ZERO {
        return Some(true);
    }
    todo!()
}
