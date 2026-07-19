use std::sync::Arc;

// use physics::PhysicsManager;
use crate::{
    inputs::Inputs,
    scene::{Level, Shape},
};

use common::Entity;
use physics::PhysicsManager;
use rendering::{RenderingManager, tex_mesh::Mesh};
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
    pub(crate) renderer_system: RenderingManager,
    physics_system: PhysicsManager,
    entities: Vec<Entity>,
    // camera: Cam3d,
    window: Arc<Window>,
    is_cursor_grabbed: bool,
}

impl Game {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let renderer_system = RenderingManager::new(&window)?;
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
            // camera, d
            window,
            is_cursor_grabbed: true,
        })
    }

    fn toggle_mouse_grab(&mut self) {
        if self.is_cursor_grabbed {
            if let Err(e) = self.window.set_cursor_grab(CursorGrabMode::None) {
                log::warn!("releasing cursor grab failed: {e}");
                return;
            }
            self.is_cursor_grabbed = false;
        } else {
            if let Err(e) = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked))
            {
                log::warn!("cursor grabbing failed: {e}");
                return;
            }
            self.is_cursor_grabbed = true;
        };
    }

    fn shape_to_mesh(shape: &Shape) -> Mesh {
        match shape {
            Shape::Rectangle { c, x, y } => Mesh::new_rectangle(
                glam::Vec3::from(*c),
                glam::Vec3::from(*x),
                glam::Vec3::from(*y),
            ),
        }
    }

    pub fn load_level(&mut self) -> anyhow::Result<()> {
        let level = Level::from_file("data/levels/2.ron")?;
        let gpu_meshes: Vec<_> = level
            .shapes
            .iter()
            .map(Self::shape_to_mesh)
            .map(|m| self.renderer_system.load_mesh(m))
            .collect::<Result<_, _>>()?;
        self.renderer_system.meshes = gpu_meshes;
        Ok(())
    }

    fn camera_move(&mut self, frame_time: u128, front: i32, right: i32, up: i32) {
        let mvmt = 0.001 * (frame_time as f32);
        self.renderer_system.camera.eye.y += up as f32 * mvmt;

        let mut dir_proj = self.renderer_system.camera.dir;
        dir_proj.y = 0.0;
        if dir_proj.x == 0.0 && dir_proj.z == 0.0 {
            dir_proj = -self.renderer_system.camera.up;
            dir_proj.y = 0.0;
        }
        let x = dir_proj.normalize();
        let y = glam::vec3(-x.z, 0.0, x.x);
        self.renderer_system.camera.eye += front as f32 * x * mvmt;
        self.renderer_system.camera.eye += right as f32 * y * mvmt;
    }

    pub fn run(&mut self, frame_time: u128, inputs: &mut Inputs) -> anyhow::Result<()> {
        let mouse_move = inputs.mouse_delta();
        if inputs.key_pressed_this_frame(PhysicalKey::Code(KeyCode::KeyG)) {
            self.toggle_mouse_grab();
        }
        if inputs.key_pressed_this_frame(PhysicalKey::Code(KeyCode::KeyR)) {
            self.load_level()
                .inspect_err(|e| log::warn!("loading level failed: {e:#}"))
                .ok();
        }
        let mut up = 0;
        let mut front = 0;
        let mut right = 0;
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::Space)) {
            up += 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyC)) {
            up -= 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyW)) {
            front += 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyS)) {
            front -= 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyD)) {
            right += 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyA)) {
            right -= 1;
        }
        self.camera_move(frame_time, front, right, up);
        if self.is_cursor_grabbed {
            self.renderer_system
                .camera
                .move_left_right(glam::Vec3::Y, -0.01 * mouse_move.0 as f32);
            self.renderer_system
                .camera
                .move_up_down(glam::Vec3::Y, 0.01 * mouse_move.1 as f32);
        }
        self.renderer_system.render()?;
        inputs.advance_frame();
        Ok(())
    }
}
