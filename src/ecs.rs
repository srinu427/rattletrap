use physics::PhysicsManager;
use vk12_rhi2::device::Device;

use crate::renderer::Renderer;

pub struct Node {
    name: String,
    entity_id: usize,
    children: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct Entity {
    renderer_id: Option<usize>,
    physics_id: Option<usize>,
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
    scene_root: Node,
    entities: Vec<Entity>,
    renderer_system: Renderer<Device>,
    renderer_infos: Vec<RenderInfo>,
    physics_system: PhysicsManager,
    physics_info: Vec<PhysicsInfo>,
}

impl EcsMega {
    pub fn run(&mut self) {
        todo!()
    }
}
