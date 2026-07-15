use std::sync::Arc;

// use physics::PhysicsManager;
use crate::inputs::Inputs;

use common::Entity;
use physics::PhysicsManager;
use rendering::RenderManager;
use winit::{
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

#[derive(Debug, Clone)]
pub struct PhysicsInfo {
    pos: glam::Vec3,
    rot: glam::Mat4,
    full_t: glam::Mat4,
    vel: glam::Vec3,
    acc: glam::Vec3,
}

pub struct Game {
    pub(crate) renderer_system: RenderManager,
    physics_system: PhysicsManager,
    entities: Vec<Entity>,
    // camera: Cam3d,
    window: Arc<Window>,
    is_cursor_grabbed: bool,
}

impl Game {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let mut renderer_system = RenderManager::new(&window)?;
        let physics_system = PhysicsManager::new();

        // let camera = Cam3d::new(
        //     glam::vec3(3., 3., 3.),
        //     glam::vec3(-1., -1., -1.),
        //     glam::Vec3::Y,
        //     2.,
        //     1.,
        // );

        Ok(Self {
            renderer_system,
            physics_system,
            entities: vec![],
            // camera,
            window,
            is_cursor_grabbed: true,
        })
    }

    fn toggle_mouse_grab(&mut self) {
        if self.is_cursor_grabbed {
            let _ = self
                .window
                .set_cursor_grab(CursorGrabMode::None)
                .inspect_err(|e| eprintln!("{e}"));
        } else {
            let _ = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked))
                .inspect_err(|e| eprintln!("{e}"));
        };
    }

    pub fn run(&mut self, frame_time: u128, inputs: &mut Inputs) -> anyhow::Result<()> {
        let mouse_move = inputs.mouse_delta();
        if inputs.key_pressed_this_frame(PhysicalKey::Code(KeyCode::KeyC)) {
            self.toggle_mouse_grab();
        }
        // self.camera
        //     .move_left_right(glam::Vec3::Y, -0.01 * mouse_move.0 as f32);
        // self.camera
        //     .move_up_down(glam::Vec3::Y, 0.01 * mouse_move.1 as f32);
        self.renderer_system.render()?;
        inputs.advance_frame();
        Ok(())
    }
}
