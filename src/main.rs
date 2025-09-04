use std::{path::Path, sync::Arc};

use winit::{
    application::ApplicationHandler, dpi::LogicalSize, event::WindowEvent, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::{Window, WindowId}
};

use renderer::{renderables::tri_mesh::{self, TriMesh}, Renderer};

#[derive(Default)]
struct App {
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_inner_size(LogicalSize { width: 800.0, height: 600.0 }))
                .unwrap(),
        );

        let mut state = Renderer::new(window.clone()).unwrap();
        state.add_mesh("square".to_string(), tri_mesh::make_square());
        state.add_texture("default".to_string(), Path::new("resources/default.png")).unwrap();
        state.add_ttpm_renderable("def".to_string(), "square".to_string(), "default".to_string()).unwrap();
        self.renderer = Some(state);

        // window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.renderer.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.draw().ok();
                // Emits a new redraw requested event.
                // state.window().request_redraw();
            }
            WindowEvent::Resized(_size) => {
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                // state.refresh_resolution()
                // .inspect_err(|e| println!("{e}"))
                // .ok();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let state = self.renderer.as_mut().unwrap();
        state.draw().ok();
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
