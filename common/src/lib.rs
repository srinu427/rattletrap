use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Entity(u64);

impl Entity {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

pub struct EntityIndices {
    pub gpu_mesh_id: Option<usize>,
    pub orientation_id: Option<usize>,
    pub kinematics_id: Option<usize>,
}

// Serde Data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Shape {
    Rectangle {
        c: [f32; 3],
        x: [f32; 3],
        y: [f32; 3],
    },
    Cube {
        c: [f32; 3],
        x: [f32; 3],
        y: [f32; 3],
        h: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsRb {
    pub mass: f32,
    pub shape: Shape,
    pub has_gravity: bool,
    pub init_location: [f32; 3],
    pub no_interact_mask: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    PhysicsRb(PhysicsRb),
}

#[derive(Serialize, Deserialize)]
pub struct Level {
    pub nodes: Vec<Node>,
}

impl Level {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let file_str = fs::read_to_string(path)?;
        let level: Self = ron::from_str(&file_str)?;
        Ok(level)
    }

    pub fn dump_to_file(&self, path: &str) -> anyhow::Result<()> {
        let data_str = ron::to_string(self)?;
        fs::write(path, data_str)?;
        Ok(())
    }
}
