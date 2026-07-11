use std::sync::Arc;

// use physics::PhysicsManager;
use crate::inputs::Inputs;
use renderer::{MeshDrawInfo, Renderer, camera::Cam3d, mesh::MeshCreateInfo};

use avk12::device::Instance;
use common::Entity;
use indexmap::IndexMap;
use physics::{
    Kinematics, PhysicsManager, RigidBody, collision_shape::CollisionShape, orient::Orientation,
};
use rendering::RenderingManager;
use winit::window::Window;
mod game_object;

#[derive(Debug, Clone)]
pub struct PhysicsInfo {
    pos: glam::Vec3,
    rot: glam::Mat4,
    full_t: glam::Mat4,
    vel: glam::Vec3,
    acc: glam::Vec3,
}

pub struct Game {
    pub(crate) renderer_system: RenderingManager,
    physics_system: PhysicsManager,
    entities: Vec<Entity>,
    camera: Cam3d,
}

impl Game {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let mut renderer_system = RenderingManager::new(&window)?;
        let physics_system = PhysicsManager::new();

        let camera = Cam3d::new(
            glam::vec3(3., 3., 3.),
            glam::vec3(-1., -1., -1.),
            glam::Vec3::Y,
            2.,
            1.,
        );

        Ok(Self {
            renderer_system,
            physics_system,
            entities: vec![],
            camera,
        })
    }

    pub fn run(&mut self, frame_time: u128, inputs: &mut Inputs) -> anyhow::Result<()> {
        let mouse_move = inputs.mouse_delta();

        self.camera
            .move_left_right(glam::Vec3::Y, -0.01 * mouse_move.0 as f32);
        self.camera
            .move_up_down(glam::Vec3::Y, 0.01 * mouse_move.1 as f32);
        self.renderer_system.render()?;
        inputs.advance_frame();
        Ok(())
    }
}
