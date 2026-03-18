mod inputs;
mod renderer;
mod renderer2;
use std::{sync::Arc, time};

use physics::PhysicsManager;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::KeyCode,
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};

use crate::{
    inputs::Inputs,
    renderer::{Renderer, level},
};

struct App {
    renderer: Option<Renderer>,
    start_time: time::Instant,
    last_frame_time_ms: u128,
    inputs: Inputs,
    physics_manager: PhysicsManager,
}

impl App {
    pub fn new() -> Self {
        Self {
            renderer: None,
            start_time: time::Instant::now(),
            last_frame_time_ms: 0,
            inputs: Inputs::new(),
            physics_manager: PhysicsManager::new(),
        }
    }

    pub fn load_level(&mut self) {
        let state = self.renderer.as_mut().unwrap();
        state.clear_meshes();
        let level = level::parse_lvl_ron("data/levels/1.ron").unwrap();
        let meshes: Vec<_> = level.geometry.iter().map(|g| g.to_mesh()).collect();
        state.add_meshes(meshes);
        let textures: Vec<_> = level.draws.iter().map(|d| d.material.as_str()).collect();
        state.add_materials(textures).unwrap();
        state.clear_mesh_draws();
        for d_info in &level.draws {
            state.add_mesh_draw_info(d_info.geo_name.clone(), d_info.material.clone());
        }
        self.physics_manager.clear();
        for geo in &level.geometry {
            self.physics_manager.add_obj(&geo.name, geo.to_rigid_body());
        }
    }
}

impl App {
    fn draw_frame(&mut self) {
        let last_frame_time = self.last_frame_time_ms;
        let current_time = self.start_time.elapsed().as_millis();
        self.last_frame_time_ms = current_time;
        let frame_time = current_time - last_frame_time;
        if self
            .inputs
            .key_pressed_this_frame(PhysicalKey::Code(KeyCode::KeyR))
        {
            println!("refreshing geo");
            self.load_level();
        }
        if self
            .inputs
            .key_pressed_this_frame(PhysicalKey::Code(KeyCode::Space))
        {
            if let Some(cube_mut) = self.physics_manager.get_obj_mut("cube") {
                cube_mut.kinematics.velocity.y = 5.0;
            };
        }
        for _ in 0..frame_time {
            self.physics_manager.forward_ms();
        }
        let state = self.renderer.as_mut().unwrap();
        for (name, id) in &self.physics_manager.object_ids {
            state.update_mesh_transform(
                name,
                self.physics_manager.objects[*id].orient.to_transform(),
            );
        }

        state.render().inspect_err(|e| eprintln!("{e}")).ok();
        self.inputs.advance_frame();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window object
        let window = event_loop
            .create_window(Window::default_attributes().with_inner_size(LogicalSize {
                width: 800.0,
                height: 600.0,
            }))
            .map(Arc::new)
            .unwrap();
        let state = Renderer::new(window).unwrap();
        self.renderer = Some(state);
        self.load_level();
        // window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.draw_frame();
            }
            WindowEvent::Resized(size) => {
                let state = self.renderer.as_mut().unwrap();
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(size, true).unwrap();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.inputs.add_key_event(event.physical_key, event.state);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.inputs.add_mouse_delta(delta);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.draw_frame();
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
