mod renderer;
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::renderer::{Renderer, mesh::Mesh};

#[derive(Default)]
struct App {
    renderer: Option<Renderer>,
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
        state.add_meshes(vec![
            Mesh::rect_cuv(
                "rectangle",
                glam::vec3(0.0, 0.0, 0.0),
                glam::vec3(0.2, 0.0, 0.0),
                glam::vec3(0.0, 0.1, 0.0),
            ),
            Mesh::rect_cuv(
                "rectangle2",
                glam::vec3(0.3, 0.0, 0.0),
                glam::vec3(0.2, 0.0, 0.0),
                glam::vec3(0.0, 0.2, 0.0),
            ),
        ]);
        self.renderer = Some(state);

        // window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let state = self.renderer.as_mut().unwrap();
                state.render().inspect_err(|e| eprintln!("{e}")).ok();
                // state.draw().ok();
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
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let state = self.renderer.as_mut().unwrap();
        state.render().inspect_err(|e| eprintln!("{e}")).ok();
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
