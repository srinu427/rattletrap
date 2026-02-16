use std::{fs, sync::Arc};

use physics::{
    Kinematics, RigidBody,
    collision_shape::{CollisionShape, Orientation, convex_mesh::ConvexMesh},
};
use serde::{Deserialize, Serialize};

use crate::renderer::mesh::Mesh;

#[derive(Debug, Serialize, Deserialize)]
pub enum Shape {
    Rect {
        pos: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
    },
    Cube {
        pos: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        h: f32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Geo {
    pub name: String,
    pub shape: Shape,
    pub has_gravity: bool,
}

impl Geo {
    pub fn to_mesh(&self) -> Mesh {
        match &self.shape {
            Shape::Rect { pos, u, v } => Mesh::rect_cuv(
                &self.name,
                glam::Vec3::from_array(pos.clone()),
                glam::Vec3::from_array(u.clone()),
                glam::Vec3::from_array(v.clone()),
            ),
            Shape::Cube { pos, u, v, h } => Mesh::cube_cuvh(
                &self.name,
                glam::Vec3::from_array(pos.clone()),
                glam::Vec3::from_array(u.clone()),
                glam::Vec3::from_array(v.clone()),
                *h,
            ),
        }
    }

    pub fn to_rigid_body(&self) -> RigidBody {
        let cm = match &self.shape {
            Shape::Rect { pos, u, v } => ConvexMesh::new_rect(
                glam::Vec3::from_array(pos.clone()),
                glam::Vec3::from_array(u.clone()),
                glam::Vec3::from_array(v.clone()),
            ),
            Shape::Cube { pos, u, v, h } => ConvexMesh::new_cube(
                glam::Vec3::from_array(pos.clone()),
                glam::Vec3::from_array(u.clone()),
                glam::Vec3::from_array(v.clone()),
                *h,
            ),
        };
        RigidBody {
            mass: 0.0,
            shape: Arc::new(CollisionShape::Mesh(cm.clone())),
            orient: Orientation::new(),
            orient_shape: CollisionShape::Mesh(cm),
            kinematics: Kinematics::new(),
            can_rotate: false,
            has_gravity: self.has_gravity,
            dont_interact_mask: 0,
            stuck: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeoDrawTarget {
    pub geo_name: String,
    pub material: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Level {
    pub geometry: Vec<Geo>,
    pub draws: Vec<GeoDrawTarget>,
}

pub fn parse_lvl_ron(path: &str) -> anyhow::Result<Level> {
    let file_str = fs::read_to_string(path)?;
    let level: Level = ron::from_str(&file_str)?;
    Ok(level)
}
