use std::sync::Arc;

use anyhow::Context;
use common::Entity;
use winit::window::Window;

use crate::{
    mesh::{Mesh, Vertex},
    mesh_pipeline::{MeshGpu, MeshRenderData},
};

mod mesh;
mod mesh_pipeline;

pub struct Renderer {
    mesh_pipeline: MeshRenderData,
    surface: wgpu::Surface<'static>,
    surface_conf: wgpu::SurfaceConfiguration,
    queue: wgpu::Queue,
    device: wgpu::Device,
    instance: wgpu::Instance,
    window: Arc<Window>,
}

impl Renderer {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Default::default(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Default::default(),
        });
        let surface = instance.create_surface(window.clone())?;
        let adapters =
            futures_executor::block_on(instance.enumerate_adapters(wgpu::Backends::default()));
        let adapter = match adapters
            .iter()
            .filter(|a| a.is_surface_supported(&surface))
            .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        {
            Some(a) => a,
            None => match adapters.get(0) {
                Some(a) => a,
                None => return Err(anyhow::Error::msg("No valid GPUs found")),
            },
        };
        let surface_caps = surface.get_capabilities(adapter);
        let surface_fmt =
            select_format(&surface_caps.formats).context("No valid surface formats found")?;
        let present_mode = surface_caps
            .present_modes
            .iter()
            .find(|p| **p == wgpu::PresentMode::Mailbox)
            .cloned()
            .unwrap_or(wgpu::PresentMode::AutoVsync);
        let window_res = window.inner_size();
        let surface_conf = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_fmt,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: window_res.width,
            height: window_res.height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        let (device, queue) =
            futures_executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
        surface.configure(&device, &surface_conf);
        let mesh_pipeline = MeshRenderData::new(&device, surface_fmt)?;
        Ok(Self {
            mesh_pipeline,
            surface,
            surface_conf,
            queue,
            device,
            instance,
            window: window.clone(),
        })
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        let new_size = self.window.inner_size();
        self.surface_conf.width = new_size.width;
        self.surface_conf.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_conf);
        Ok(())
    }

    pub fn render(&mut self) {}

    pub fn load_mesh_data(&mut self, entity: Entity, mesh: Mesh) -> anyhow::Result<()> {
        let vb_size = mesh.verts.len() * size_of::<Vertex>();
        let vb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{entity:?}_vb")),
            size: vb_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&vb, 0, bytemuck::cast_slice(&mesh.verts));
        let ib_size = mesh.idxs.len() * size_of::<u16>();
        let ib = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{entity:?}_ib")),
            size: ib_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&ib, 0, bytemuck::cast_slice(&mesh.idxs));
        self.mesh_pipeline.meshes.insert(
            entity,
            MeshGpu {
                vb,
                ib,
                indx_count: mesh.idxs.len() as u32,
            },
        );
        Ok(())
    }
}

fn select_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    let mut surface_format = None;
    for format in formats {
        match format {
            wgpu::TextureFormat::Rgb10a2Unorm | wgpu::TextureFormat::Rgba16Float => {
                surface_format = Some(*format)
            }
            _ => {}
        }
    }
    if surface_format.is_none() {
        for format in formats {
            match format {
                wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb => {
                    surface_format = Some(*format)
                }
                _ => {}
            }
        }
    }
    if surface_format.is_none() {
        for format in formats {
            match format {
                wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb => {
                    surface_format = Some(*format)
                }
                _ => {}
            }
        }
    }
    surface_format
}
