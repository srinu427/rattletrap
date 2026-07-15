use std::sync::Arc;

use winit::window::Window;

use crate::model::{MeshPipeline, ModelGpu};

pub mod model;

pub struct RenderManager {
    model_gpu: ModelGpu,
    mesh_pipeline: MeshPipeline,
    queue: wgpu::Queue,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    surface_config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'static>,
    window: Arc<Window>,
}

impl RenderManager {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let surface = instance.create_surface(window.clone())?;
        let adapter =
            futures_executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: true,
            }))?;
        let (device, queue) =
            futures_executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }))?;
        let surface_caps = surface.get_capabilities(&adapter);
        let (surface_fmt, color_space) = choose_surface_fmt(&surface_caps.formats);
        let window_res = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_fmt,
            width: window_res.width,
            height: window_res.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: color_space,
        };
        surface.configure(&device, &surface_config);
        let mesh_pipeline = MeshPipeline::new(&device, surface_fmt);
        let model_gpu = ModelGpu::new(&device, &queue);
        Ok(Self {
            model_gpu,
            mesh_pipeline,
            surface,
            device,
            queue,
            adapter,
            surface_config,
            window: window.clone(),
        })
    }

    pub fn resize(&mut self) {
        let window_res = self.window.inner_size();
        let surface_caps = self.surface.get_capabilities(&self.adapter);
        let (surface_fmt, color_space) = choose_surface_fmt(&surface_caps.formats);
        if window_res.width > 0 && window_res.height > 0 {
            self.surface_config.format = surface_fmt;
            self.surface_config.color_space = color_space;
            self.surface_config.width = window_res.width;
            self.surface_config.height = window_res.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        // self.window.request_redraw();

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip this frame
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                self.resize();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // You could recreate the devices and all resources
                // created with it here, but we'll just bail
                anyhow::bail!("Lost device");
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.mesh_pipeline.pipeline);
            render_pass.set_vertex_buffer(0, self.model_gpu.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.model_gpu.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw(0..self.model_gpu.index_len, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify();
        self.queue.present(output);

        Ok(())
    }
}

static SURF_FMT_PREF: &[(wgpu::TextureFormat, wgpu::SurfaceColorSpace); 4] = &[
    (
        wgpu::TextureFormat::Rgba16Float,
        wgpu::SurfaceColorSpace::ExtendedSrgbLinear,
    ),
    (
        wgpu::TextureFormat::Rgb10a2Unorm,
        wgpu::SurfaceColorSpace::Bt2100Pq,
    ),
    (
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::SurfaceColorSpace::Srgb,
    ),
    (
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::SurfaceColorSpace::Srgb,
    ),
];

fn choose_surface_fmt(
    supported_fmts: &Vec<wgpu::TextureFormat>,
) -> (wgpu::TextureFormat, wgpu::SurfaceColorSpace) {
    for (fmt, cs) in SURF_FMT_PREF {
        if supported_fmts.contains(fmt) {
            return (*fmt, *cs);
        }
    }
    (supported_fmts[0], wgpu::SurfaceColorSpace::Auto)
}
