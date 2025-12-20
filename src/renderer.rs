use std::sync::Arc;

use image::EncodableLayout;
use include_bytes_aligned::include_bytes_aligned;
use rhi::enumflags2::BitFlags;
use winit::window::Window;

use crate::renderer::{camera::Cam3d, mesh::Vertex};

mod camera;
mod mesh;

static VERT_SPV: &[u8] = include_bytes_aligned!(4, "shaders/triangle.vert.spv");
static FRAG_SPV: &[u8] = include_bytes_aligned!(4, "shaders/triangle.frag.spv");
static TRIANGLE_VERTS: &[Vertex] = &[
    Vertex {
        pos: glam::vec4(1.0, 1.0, 0.0, 1.0),
        uv: glam::vec4(1.0, 0.0, 0.0, 1.0),
    },
    Vertex {
        pos: glam::vec4(-1.0, 1.0, 0.0, 1.0),
        uv: glam::vec4(0.0, 1.0, 0.0, 1.0),
    },
    Vertex {
        pos: glam::vec4(0.0, -1.0, 0.0, 1.0),
        uv: glam::vec4(0.0, 0.0, 1.0, 1.0),
    },
];
static TRIANGLE_IDXS: &[u16] = &[0, 1, 2];

pub struct Renderer {
    window: Arc<Window>,
    swapchain: rhi::Swapchain,
    swapchain_image_initialized: Vec<bool>,
    cmd_buffers: Vec<rhi::CommandBuffer>,
    draw_sems: Vec<rhi::Semaphore>,
    draw_sem_nums: Vec<u64>,
    present_sems: Vec<rhi::Semaphore>,
    bg_image: rhi::Image,
    pipeline: rhi::RenderPipeline,
    camera: Cam3d,
    render_outputs: Vec<rhi::RenderOutput>,
    vertex_buffers: Vec<rhi::Buffer>,
    index_buffers: Vec<rhi::Buffer>,
    camera_buffers: Vec<rhi::Buffer>,
    camera_stage_buffers: Vec<rhi::Buffer>,
    camera_dsets: Vec<rhi::DSet>,
    device: rhi::Device,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let device = rhi::Device::new(&window)?;
        let swapchain = device.create_swapchain()?;
        let swapchain_image_initialized = vec![false; swapchain.images().len()];
        let draw_sems: Vec<_> = (0..swapchain.images().len())
            .map(|_| device.create_semaphore(false))
            .collect::<Result<_, _>>()?;
        let draw_sem_nums = vec![0; swapchain.images().len()];
        let present_sems: Vec<_> = (0..swapchain.images().len())
            .map(|_| device.create_semaphore(true))
            .collect::<Result<_, _>>()?;
        let cmd_buffers: Vec<_> = (0..swapchain.images().len())
            .map(|_| device.graphics_queue().create_command_buffer())
            .collect::<Result<_, _>>()?;
        let bg_image = Self::load_bg_image(&device, "./default.png")?;
        let vert_shader = device.load_shader(VERT_SPV)?;
        let frag_shader = device.load_shader(FRAG_SPV)?;
        let mut pipeline = device.create_render_pipeline(
            rhi::VertexStageInfo {
                shader: &vert_shader,
                entrypoint: "main",
                attribs: vec![rhi::VertexAttribute::Vec4, rhi::VertexAttribute::Vec4],
                stride: core::mem::size_of::<Vertex>(),
            },
            rhi::FragmentStageInfo {
                shader: &frag_shader,
                entrypoint: "main",
                outputs: vec![rhi::FragmentOutputInfo {
                    format: swapchain.images()[0].format(),
                    clear: false,
                    store: true,
                }],
            },
            rhi::RasterMode::Fill(1.0),
            vec![vec![rhi::DBindingType::UBuffer(1)]],
            0,
        )?;
        let render_outputs = swapchain
            .views()
            .iter()
            .map(|iv| pipeline.new_output(vec![iv]))
            .collect::<Result<_, _>>()?;
        let mut camera = Cam3d {
            eye: glam::vec3(1.0, 0.0, 5.0),
            fov: 120.0,
            dir: glam::vec3(0.0, 0.0, -1.0),
            aspect: 1.0,
            up: glam::vec3(0.0, 1.0, 0.0),
            padding: 0,
            proj_view: glam::Mat4::IDENTITY,
        };
        camera.update_proj_view();
        let vertex_buffers = (0..swapchain.images().len())
            .map(|_| {
                Self::gpu_buffer_w_data(
                    &device,
                    bytemuck::cast_slice(TRIANGLE_VERTS),
                    rhi::BufferFlags::Vertex.into(),
                )
            })
            .collect::<Result<_, _>>()?;
        let index_buffers = (0..swapchain.images().len())
            .map(|_| {
                Self::gpu_buffer_w_data(
                    &device,
                    bytemuck::cast_slice(TRIANGLE_IDXS),
                    rhi::BufferFlags::Index.into(),
                )
            })
            .collect::<Result<_, _>>()?;
        let camera_buffers: Vec<_> = (0..swapchain.images().len())
            .map(|_| {
                Self::gpu_buffer_w_data(
                    &device,
                    bytemuck::bytes_of(&camera),
                    rhi::BufferFlags::Uniform.into(),
                )
            })
            .collect::<Result<_, _>>()?;
        let camera_stage_buffers: Vec<_> = (0..swapchain.images().len())
            .map(|_| {
                device.create_buffer(
                    core::mem::size_of::<Cam3d>() as _,
                    rhi::BufferFlags::CopySrc.into(),
                    rhi::MemLocation::CpuToGpu,
                )
            })
            .collect::<Result<_, _>>()?;
        let mut camera_dsets: Vec<_> = (0..swapchain.images().len())
            .map(|_| pipeline.new_set(0))
            .collect::<Result<_, _>>()?;

        for i in 0..swapchain.images().len() {
            camera_dsets[i].write(vec![rhi::DBindingData::UBuffer(vec![&camera_buffers[i]])]);
        }

        Ok(Self {
            window,
            device,
            swapchain,
            swapchain_image_initialized,
            draw_sems,
            draw_sem_nums,
            cmd_buffers,
            present_sems,
            pipeline,
            camera,
            render_outputs,
            vertex_buffers,
            index_buffers,
            camera_buffers,
            camera_stage_buffers,
            camera_dsets,
            bg_image,
        })
    }

    fn load_bg_image(device: &rhi::Device, path: &str) -> anyhow::Result<rhi::Image> {
        let image_data = image::open(path)?;
        let image_data_rgba = image_data.to_rgba8();
        let image_data_bytes = image_data_rgba.as_bytes();
        let mut stage_buffer = device.create_buffer(
            image_data_bytes.len() as _,
            rhi::BufferFlags::CopySrc.into(),
            rhi::MemLocation::CpuToGpu,
        )?;
        stage_buffer.write_data(&image_data_bytes)?;
        let image = device.create_image(
            rhi::Dimension::D2,
            rhi::Format::Rgba8Srgb,
            image_data.width(),
            image_data.height(),
            1,
            1,
            rhi::ImageUsage::CopyDst | rhi::ImageUsage::CopySrc,
            rhi::MemLocation::Gpu,
        )?;
        let cmd_buffer = device.graphics_queue().create_command_buffer()?;
        let mut encoder = cmd_buffer.encoder()?;
        encoder.set_last_image_access(&image, rhi::ImageAccess::Undefined, 0..1, 0..1);
        encoder.copy_buffer_to_image(&stage_buffer, &image, 0..1, 0);
        encoder.set_last_image_access(
            &image,
            rhi::ImageAccess::Transfer(rhi::RWAccess::Read),
            0..1,
            0..1,
        );
        encoder.finalize()?;
        let semaphore = device.create_semaphore(false)?;
        cmd_buffer.submit(vec![], vec![semaphore.submit_info(1)])?;
        semaphore.wait_for(1, None)?;
        drop(stage_buffer);
        Ok(image)
    }

    fn gpu_buffer_w_data(
        device: &rhi::Device,
        data: &[u8],
        mut usage: BitFlags<rhi::BufferFlags>,
    ) -> anyhow::Result<rhi::Buffer> {
        let mut stage_buffer = device.create_buffer(
            data.len() as _,
            rhi::BufferFlags::CopySrc.into(),
            rhi::MemLocation::CpuToGpu,
        )?;
        stage_buffer.write_data(data)?;
        usage |= rhi::BufferFlags::CopyDst;
        let buffer = device.create_buffer(data.len() as _, usage, rhi::MemLocation::Gpu)?;
        let cmd_buffer = device.graphics_queue().create_command_buffer()?;
        let mut encoder = cmd_buffer.encoder()?;
        encoder.copy_buffer_to_buffer(&stage_buffer, &buffer);
        encoder.finalize()?;
        let sem = device.create_semaphore(false)?;
        cmd_buffer.submit(vec![], vec![sem.submit_info(1)])?;
        sem.wait_for(1, None)?;
        drop(stage_buffer);
        Ok(buffer)
    }

    pub fn resize(
        &mut self,
        new_size: winit::dpi::PhysicalSize<u32>,
        redraw: bool,
    ) -> anyhow::Result<()> {
        for idx in 0..self.swapchain.images().len() {
            self.draw_sems[idx]
                .wait_for(self.draw_sem_nums[idx], None)
                .inspect_err(|e| eprintln!("error waiting for running draws: {e}"))
                .ok();
        }
        self.render_outputs.clear();
        if let Err(e) = self.swapchain.resize(new_size.width, new_size.height) {
            eprintln!("resizing swapchain failed: {e}");
        } else {
            self.swapchain_image_initialized = vec![false; self.swapchain.images().len()];
        }
        self.render_outputs = self
            .swapchain
            .views()
            .iter()
            .map(|iv| self.pipeline.new_output(vec![iv]))
            .collect::<Result<_, _>>()?;
        if redraw {
            self.render()?;
        }
        self.window.request_redraw();
        Ok(())
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let (mut img_idx, is_unoptimal) = self.swapchain.acquire_image()?;
        if is_unoptimal {
            let res = self.window.inner_size();
            self.resize(res, false)?;
            (img_idx, _) = self.swapchain.acquire_image()?;
        }
        let idx = img_idx as usize;
        self.draw_sems[idx].wait_for(self.draw_sem_nums[idx], None)?;
        let aspect_ratio = self.swapchain.images()[0].width() as f32
            / self.swapchain.images()[0].height().max(1) as f32;
        self.camera.aspect = aspect_ratio;
        self.camera.update_proj_view();
        self.camera_stage_buffers[idx].write_data(bytemuck::bytes_of(&self.camera))?;
        let mut encoder = self.cmd_buffers[idx].encoder()?;
        encoder.copy_buffer_to_buffer(&self.camera_stage_buffers[idx], &self.camera_buffers[idx]);
        if self.swapchain_image_initialized[idx] {
            encoder.set_last_image_access(
                &self.swapchain.images()[idx],
                rhi::ImageAccess::Present,
                0..1,
                0..1,
            );
        } else {
            encoder.set_last_image_access(
                &self.swapchain.images()[idx],
                rhi::ImageAccess::Undefined,
                0..1,
                0..1,
            );
        }
        encoder.blit_image_2d_stretch(&self.bg_image, &self.swapchain.images()[idx], 0, 0);

        let mut render_pass = encoder.start_render_pipeline(
            &self.pipeline,
            &self.render_outputs[idx],
            vec![rhi::ClearValue::Colour([1.0; 4])],
        );
        render_pass.bind_vbs(vec![&self.vertex_buffers[idx]]);
        render_pass.bind_ib(&self.index_buffers[idx], rhi::IndexType::U16);
        render_pass.bind_dsets(vec![&self.camera_dsets[idx]]);
        render_pass.draw_indexed(TRIANGLE_IDXS.len() as _);
        let mut encoder = render_pass.end();
        encoder.set_last_image_access(
            &self.swapchain.images()[idx],
            rhi::ImageAccess::Present,
            0..1,
            0..1,
        );
        encoder.finalize()?;
        self.draw_sem_nums[idx] += 1;
        self.cmd_buffers[idx].submit(
            vec![],
            vec![
                self.draw_sems[idx].submit_info(self.draw_sem_nums[idx]),
                self.present_sems[idx].submit_info(1),
            ],
        )?;

        self.window.pre_present_notify();
        self.swapchain
            .present_image(img_idx, &self.present_sems[idx])?;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        for idx in 0..self.swapchain.images().len() {
            self.draw_sems[idx]
                .wait_for(self.draw_sem_nums[idx], None)
                .inspect_err(|e| eprintln!("error waiting for running draws: {e}"))
                .ok();
        }
        self.device.graphics_queue().wait_idle();
    }
}
