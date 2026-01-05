mod inputs;
mod renderer;
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::KeyCode,
    window::{Window, WindowId},
};

use crate::{
    inputs::Inputs,
    renderer::{Renderer, level},
};

struct App {
    renderer: Option<Renderer>,
    inputs: Inputs,
}

impl App {
    pub fn new() -> Self {
        Self {
            renderer: None,
            inputs: Inputs::new(),
        }
    }
}

impl App {
    fn draw_frame(&mut self) {
        let state = self.renderer.as_mut().unwrap();
        if self
            .inputs
            .key_pressed_this_frame(winit::keyboard::PhysicalKey::Code(KeyCode::KeyR))
        {
            println!("refreshing geo");
            state.clear_meshes();
            let meshes = level::parse_lvl("data/levels/1.lvl").unwrap();
            state.add_meshes(meshes);
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
        let mut state = Renderer::new(window).unwrap();
        let meshes = level::parse_lvl("data/levels/1.lvl").unwrap();
        state.add_meshes(meshes);
        self.renderer = Some(state);

        // window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.draw_frame();
                // Emits a new redraw requested event.
                // state.window().request_redraw();
            }
            WindowEvent::Resized(size) => {
                let state = self.renderer.as_mut().unwrap();
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(size, true).unwrap();
                // state.render().inspect_err(|e| eprintln!("{e}")).ok();
                // state.refresh_resolution()
                // .inspect_err(|e| println!("{e}"))
                // .ok();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.inputs.add_key_event(event.physical_key, event.state);
                match event.physical_key {
                    winit::keyboard::PhysicalKey::Code(key_code) => if key_code == KeyCode::KeyR {},
                    winit::keyboard::PhysicalKey::Unidentified(_native_key_code) => todo!(),
                }
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
