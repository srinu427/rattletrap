use std::{hash::Hash, sync::Arc};

// use physics::PhysicsManager;
use crate::renderer::{MeshDrawInfo, Renderer, mesh::MeshCreateInfo};
use avk12::device::Instance;
use enumflags2::bitflags;
use hashbrown::HashMap;
use winit::window::Window;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Entity(u64);

pub struct Node {
    name: String,
    entity: Entity,
    children: Vec<Self>,
}

#[bitflags]
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum Component {
    Renderer,
}

pub struct VBMap<K: Hash + Eq, T> {
    entity_id_map: HashMap<K, usize>,
    data: Vec<T>,
}

impl<K, T> VBMap<K, T>
where
    K: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            entity_id_map: HashMap::new(),
            data: Vec::new(),
        }
    }

    pub fn add(&mut self, key: K, data: T) {
        match self.entity_id_map.get(&key) {
            Some(&id) => {
                self.data[id] = data;
            }
            None => {
                let ins_idx = self.data.len();
                self.data.push(data);
                self.entity_id_map.insert(key, ins_idx);
            }
        }
    }

    pub fn get(&self, key: &K) -> Option<&T> {
        let idx = self.entity_id_map.get(key)?.clone();
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut T> {
        let idx = self.entity_id_map.get(key)?.clone();
        self.data.get_mut(idx)
    }

    pub fn remove(&mut self, key: K) {
        if let Some(idx) = self.entity_id_map.get(&key).cloned() {
            let last_idx = self.data.len() - 1;
            if idx != last_idx {
                self.data.swap(idx, last_idx);
            }
            self.data.pop();
        }
    }

    pub fn raw_data(&self) -> &Vec<T> {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct RenderInfo {
    mesh: usize,
    texture: String,
    drawable_id: usize,
}

#[derive(Debug, Clone)]
pub struct PhysicsInfo {
    pos: glam::Vec3,
    rot: glam::Mat4,
    full_t: glam::Mat4,
    vel: glam::Vec3,
    acc: glam::Vec3,
}

pub struct EcsMega {
    // scene_root: Node,
    entities: Vec<Entity>,
    pub(crate) renderer_system: Renderer,
    mesh_draw_infos: Vec<MeshDrawInfo>,
    // physics_system: PhysicsManager,
    // physics_info: Vec<PhysicsInfo>,
}

impl EcsMega {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let inst = Instance::new(&window)?;
        let device = inst.init_device(0)?;
        let mut renderer_system = Renderer::new(device)?;
        let mesh_draw = renderer_system.load_mesh_draw(
            "rect1".to_string(),
            MeshCreateInfo::RectCUV {
                c: [0.0; 3],
                u: [0.75, 0.0, 0.0],
                v: [0.0, 0.75, 0.0],
            },
            "data/textures/default.png".to_string(),
            true,
        )?;
        Ok(Self {
            entities: vec![],
            renderer_system,
            mesh_draw_infos: vec![mesh_draw],
        })
    }

    pub fn run(&mut self, frame_time: u128) -> anyhow::Result<()> {
        self.renderer_system.render(&self.mesh_draw_infos)?;
        Ok(())
    }
}
