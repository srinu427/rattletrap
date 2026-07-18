use std::sync::Arc;

use winit::window::Window;

use crate::{
    tex_mesh::TexMeshPass,
    vkraii::{
        command::CommandBufferRaii,
        device::DeviceRaii,
        resource::{BufferRaii, ImageAccess, ImageRaii, ImageViewKey},
        swapchain::SwapchainRaii,
    },
};

pub mod model;

pub struct RenderingManager {
    textures: IndexMap<String, ImageRaii>,
    pipeline: TexMeshPass,
    deferred_cb: Option<CommandBufferRaii>,
    swapchain: SwapchainRaii,
    device: DeviceRaii,
    window: Arc<Window>,
}

impl RenderManager {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let mut device = DeviceRaii::new(window)?;
        let swapchain = SwapchainRaii::new(&device.device_d)?;
        let mut deferred_cb = device.command_pool.get_cb()?;
        deferred_cb.begin()?;
        let pipeline = TexMeshPass::new(&mut device, swapchain.format, &mut deferred_cb)?;
        Ok(Self {
            textures: Default::default(),
            pipeline,
            deferred_cb: Some(deferred_cb),
            swapchain,
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

    pub fn render(&mut self) -> anyhow::Result<()> {
        let Some(mut curr_frame) = self.swapchain.acquire_image()? else {
            self.refresh_size()?;
            return Ok(());
        };
        if let Some(deferred_cb) = self.deferred_cb.take() {
            let task = self.device.run_commands(vec![deferred_cb])?;
            self.device.wait_on_task(task)?;
        }
        let mut command_buffer = self.device.command_pool.get_cb()?;
        command_buffer.begin()?;
        if curr_frame.get_image().access.layout != vk::ImageLayout::PRESENT_SRC_KHR {
            curr_frame.get_image().barrier(
                command_buffer.command_buffer,
                ImageAccess {
                    access_flags: vk::AccessFlags::MEMORY_READ,
                    layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                },
                0..1,
                0..1,
            );
        }
        let view = curr_frame.get_image().get_view(&ImageViewKey {
            type_: vk::ImageViewType::TYPE_2D,
            layer_range: 0..1,
            level_range: 0..1,
        })?;
        self.pipeline.begin(
            command_buffer.command_buffer,
            (curr_frame.get_image().res.0, curr_frame.get_image().res.1),
            vec![view],
        )?;
        unsafe {
            self.device.device_d.device.cmd_bind_vertex_buffers(
                command_buffer.command_buffer,
                0,
                &[self.pipeline.gpu_mesh.vertex_buffer.buffer],
                &[0],
            );
            self.device.device_d.device.cmd_bind_index_buffer(
                command_buffer.command_buffer,
                self.pipeline.gpu_mesh.index_buffer.buffer,
                0,
                vk::IndexType::UINT16,
            );
            self.device
                .device_d
                .device
                .cmd_draw(command_buffer.command_buffer, 3, 1, 0, 0);
        }
        self.pipeline.end(command_buffer.command_buffer);
        let task = self.device.run_commands(vec![command_buffer])?;
        self.device.wait_on_task(task)?;
        self.device
            .device_d
            .instance_raii
            .window
            .pre_present_notify();
        drop(curr_frame);
        Ok(())
    }
}
