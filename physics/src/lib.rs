use crate::collision_shape::{CollisionShape, Orientation};

pub mod collision_shape;

#[derive(Debug, Clone)]
pub struct Kinematics {
    velocity: glam::Vec3,
    acceleration: glam::Vec3,
}

#[derive(Debug, Clone)]
pub struct RigidBody {
    name: String,
    shape: CollisionShape,
    orientation: Orientation,
    kinematics: Kinematics,
}

pub fn run_physics_sim(rigid_bodies: &mut [RigidBody]) {
    let moved_coll_shapes: Vec<_> = rigid_bodies
        .iter()
        .map(|rb| rb.shape.with_orientation(&rb.orientation))
        .collect();
    let distances: Vec<Vec<_>> = (0..rigid_bodies.len())
        .map(|i| {
            (i + 1..rigid_bodies.len())
                .map(|j| CollisionShape::min_distance(&moved_coll_shapes[i], &moved_coll_shapes[j]))
                .collect()
        })
        .collect();

    for i in 0..rigid_bodies.len() {
        let min_dist = (0..rigid_bodies.len())
            .map(|j| {
                if i < j {
                    distances[i][j - i - 1]
                } else if i > j {
                    distances[j][i - j - 1]
                } else {
                    (f32::MAX, glam::Vec3::ZERO)
                }
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
    }

    todo!()
}
