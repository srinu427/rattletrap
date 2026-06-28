use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Entity(u64);

impl Entity {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometryInfoDisk {
    RectCUV {
        c: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
    },
    CubeCUVH {
        c: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        h: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshInfoDisk {
    Geo(GeometryInfoDisk),
    File(String),
    UsePhysicsShape,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsInfoDisk {
    RigidBody {
        mass: f32,
        shape: GeometryInfoDisk,
        gravity: bool,
        #[serde(default)]
        init_velocity: [f32; 3],
        #[serde(default)]
        init_acceleration: [f32; 3],
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfoDisk {
    name: String,
    physics: Option<PhysicsInfoDisk>,
    mesh: Option<MeshInfoDisk>,
    transform: Option<[[f32; 4]; 4]>,
}

pub struct GameData {
    names: IndexMap<Entity, String>,
}
