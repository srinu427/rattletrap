use std::{hash::Hash, sync::Arc};

// use physics::PhysicsManager;
use crate::{
    inputs::Inputs,
    renderer::{MeshDrawInfo, Renderer, camera::Cam3d, mesh::MeshCreateInfo},
};
use avk12::device::Instance;
use hashbrown::HashMap;
use physics::{PhysicsManager, RigidBody};
use winit::window::Window;
mod game_object;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Entity(u64);

pub struct Node {
    name: String,
    entity: Entity,
    children: Vec<Self>,
}

pub struct VBMap<K: Hash + Eq, T> {
    id_map: HashMap<K, usize>,
    data: Vec<T>,
}

impl<K, T> VBMap<K, T>
where
    K: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            id_map: HashMap::new(),
            data: Vec::new(),
        }
    }

    pub fn add(&mut self, key: K, data: T) {
        match self.id_map.get(&key) {
            Some(&id) => {
                self.data[id] = data;
            }
            None => {
                let ins_idx = self.data.len();
                self.data.push(data);
                self.id_map.insert(key, ins_idx);
            }
        }
    }

    pub fn get(&self, key: &K) -> Option<&T> {
        let idx = self.id_map.get(key)?.clone();
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut T> {
        let idx = self.id_map.get(key)?.clone();
        self.data.get_mut(idx)
    }

    pub fn remove(&mut self, key: K) {
        if let Some(idx) = self.id_map.get(&key).cloned() {
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
pub struct PhysicsInfo {
    pos: glam::Vec3,
    rot: glam::Mat4,
    full_t: glam::Mat4,
    vel: glam::Vec3,
    acc: glam::Vec3,
}

pub struct Game {
    pub(crate) renderer_system: Renderer,
    physics_system: PhysicsManager,
    entities: Vec<Entity>,
    mesh_draw_infos: VBMap<Entity, MeshDrawInfo>,
    camera: Cam3d,
    rigid_bodies: VBMap<Entity, RigidBody>,
}

impl Game {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let inst = Instance::new(&window)?;
        let device = inst.init_device(0)?;
        let mut renderer_system = Renderer::new(device)?;
        let mesh_draw = renderer_system.load_mesh_draw(
            "rect1".to_string(),
            MeshCreateInfo::RectCUV {
                c: [0.; 3],
                u: [3., 0., 0.],
                v: [0., 3., 0.],
            },
            "data/textures/default.png".to_string(),
            true,
        )?;
        let camera = Cam3d::new(
            glam::vec3(3., 3., 3.),
            glam::vec3(-1., -1., -1.),
            glam::Vec3::Y,
            2.,
            1.,
        );
        let entity = Entity(0);
        let mut mesh_draw_infos = VBMap::new();
        mesh_draw_infos.add(entity, mesh_draw);
        let physics_system = PhysicsManager::new();
        Ok(Self {
            renderer_system,
            physics_system,
            entities: vec![entity],
            mesh_draw_infos,
            camera,
            rigid_bodies: VBMap::new(),
        })
    }

    pub fn run(&mut self, frame_time: u128, inputs: &mut Inputs) -> anyhow::Result<()> {
        let mouse_move = inputs.mouse_delta();
        self.camera
            .move_left_right(glam::Vec3::Y, -0.01 * mouse_move.0 as f32);
        self.camera
            .move_up_down(glam::Vec3::Y, 0.01 * mouse_move.1 as f32);
        self.renderer_system
            .render(&mut self.camera, &self.mesh_draw_infos.raw_data())?;
        inputs.advance_frame();
        Ok(())
    }
}
