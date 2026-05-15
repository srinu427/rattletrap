mod ecs;
mod inputs;
mod renderer;
mod scene;
use std::{sync::Arc, time};

// use physics::PhysicsManager;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::{ecs::EcsMega, inputs::Inputs};

struct App {
    ecs_mega: Option<EcsMega>,
    start_time: time::Instant,
    last_frame_time_ms: u128,
    inputs: Inputs,
}

impl App {
    pub fn new() -> Self {
        Self {
            ecs_mega: None,
            start_time: time::Instant::now(),
            last_frame_time_ms: 0,
            inputs: Inputs::new(),
        }
    }
}

impl App {
    fn draw_frame(&mut self) {
        let last_frame_time = self.last_frame_time_ms;
        let current_time = self.start_time.elapsed().as_millis();
        self.last_frame_time_ms = current_time;
        let frame_time = current_time - last_frame_time;

        let ecs_mut = self.ecs_mega.as_mut().unwrap();
        if let Err(e) = ecs_mut.run(frame_time, &mut self.inputs) {
            eprintln!("failure running ECS: {e}");
        }
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
        let _ = window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
        self.ecs_mega = Some(EcsMega::new(window).unwrap());
        // self.load_level();
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
            WindowEvent::Resized(_size) => {
                let Some(ecs_mut) = self.ecs_mega.as_mut() else {
                    return;
                };
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                if let Err(e) = ecs_mut.renderer_system.resize() {
                    eprintln!("error resizing rendering system: {e}");
                }
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
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
