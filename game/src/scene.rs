use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Shape {
    Rectangle {
        c: [f32; 3],
        x: [f32; 3],
        y: [f32; 3],
    },
}

#[derive(Serialize, Deserialize)]
pub struct Level {
    pub shapes: Vec<Shape>,
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
