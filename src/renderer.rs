use std::sync::Arc;

use hashbrown::HashMap;
use image::EncodableLayout;
use include_bytes_aligned::include_bytes_aligned;
use winit::window::Window;

use crate::renderer::{
    camera::Cam3d,
    mesh::{Mesh, Vertex},
};

mod camera;
pub mod level;
pub mod mesh;

static VERT_SPV: &[u8] = include_bytes_aligned!(4, "shaders/triangle.vert.spv");
static FRAG_SPV: &[u8] = include_bytes_aligned!(4, "shaders/triangle.frag.spv");

#[derive(Debug, Clone)]
pub struct MeshDrawParams {
    name: String,
    vb_offset: usize,
    ib_offset: usize,
    len: usize,
}

pub struct PerFrameData {
    index: usize,
    meshes: HashMap<String, Arc<Mesh>>,
    mesh_offsets: HashMap<String, MeshDrawParams>,
    vb_up_to_date: u32,
    swapchain_image_initialized: u32,
    cmd_buffer: rhi::CommandBuffer,
    draw_sem: rhi::Semaphore,
    draw_sem_num: u64,
    present_sem: rhi::Semaphore,
    vertex_buffer: rhi::Buffer,
    vertex_stage_buffer: rhi::Buffer,
    index_buffer: rhi::Buffer,
    index_stage_buffer: rhi::Buffer,
    camera_buffer: rhi::Buffer,
    camera_stage_buffer: rhi::Buffer,
    camera_dset: rhi::DSet,
}

impl PerFrameData {
    pub fn new(
        device: &rhi::Device,
        pipeline: &mut rhi::RenderPipeline,
        index: usize,
    ) -> anyhow::Result<Self> {
        let cmd_buffer = device.graphics_queue().create_command_buffer()?;
        let draw_sem = device.create_semaphore(false)?;
        let present_sem = device.create_semaphore(true)?;
        let vertex_buffer = device.create_buffer(
            32 * 1024 * 1024,
            rhi::BufferFlags::Vertex | rhi::BufferFlags::CopyDst,
            rhi::MemLocation::Gpu,
        )?;
        let vertex_stage_buffer = device.create_buffer(
            32 * 1024 * 1024,
            rhi::BufferFlags::CopySrc.into(),
            rhi::MemLocation::CpuToGpu,
        )?;
        let index_buffer = device.create_buffer(
            1 * 1024 * 1024,
            rhi::BufferFlags::Index | rhi::BufferFlags::CopyDst,
            rhi::MemLocation::Gpu,
        )?;
        let index_stage_buffer = device.create_buffer(
            1 * 1024 * 1024,
            rhi::BufferFlags::CopySrc.into(),
            rhi::MemLocation::CpuToGpu,
        )?;
        let camera_buffer = device.create_buffer(
            core::mem::size_of::<Cam3d>() as _,
            rhi::BufferFlags::Uniform | rhi::BufferFlags::CopyDst,
            rhi::MemLocation::Gpu,
        )?;
        let camera_stage_buffer = device.create_buffer(
            core::mem::size_of::<Cam3d>() as _,
            rhi::BufferFlags::CopySrc.into(),
            rhi::MemLocation::CpuToGpu,
        )?;
        let mut camera_dset = pipeline.new_set(0)?;
        camera_dset.write(vec![rhi::DBindingData::UBuffer(vec![&camera_buffer])]);
        Ok(Self {
            index,
            meshes: HashMap::new(),
            mesh_offsets: HashMap::new(),
            vb_up_to_date: 0,
            swapchain_image_initialized: 0,
            cmd_buffer,
            draw_sem,
            draw_sem_num: 0,
            present_sem,
            vertex_buffer,
            vertex_stage_buffer,
            index_buffer,
            index_stage_buffer,
            camera_buffer,
            camera_stage_buffer,
            camera_dset,
        })
    }

    pub fn add_meshes(&mut self, meshes: &Vec<Arc<Mesh>>) {
        for mesh in meshes {
            self.meshes.insert(mesh.name.clone(), mesh.clone());
        }
        self.vb_up_to_date = 0;
    }

    pub fn clear_meshes(&mut self) {
        self.meshes.clear();
        self.vb_up_to_date = 0;
    }

    pub fn update_vb(&mut self, encoder: &mut rhi::CommandEncoder) -> anyhow::Result<()> {
        if self.vb_up_to_date > 0 {
            return Ok(());
        }
        let mut all_verts = vec![];
        let mut all_inds = vec![];
        self.mesh_offsets.clear();
        for (name, mesh_info) in &self.meshes {
            let vb_offset = all_verts.len();
            let ib_offset = all_inds.len();
            let ind_len = mesh_info.idxs.len();
            all_verts.extend(mesh_info.verts.clone());
            all_inds.extend(mesh_info.idxs.clone());
            self.mesh_offsets.insert(
                name.clone(),
                MeshDrawParams {
                    name: name.clone(),
                    vb_offset,
                    ib_offset,
                    len: ind_len,
                },
            );
        }
        let vert_bytes = bytemuck::cast_slice(&all_verts);
        let ind_bytes = bytemuck::cast_slice(&all_inds);
        self.vertex_stage_buffer.write_data(vert_bytes)?;
        self.index_stage_buffer.write_data(ind_bytes)?;
        encoder.copy_buffer_to_buffer(
            &self.vertex_stage_buffer,
            &self.vertex_buffer,
            Some(vert_bytes.len() as _),
        );
        encoder.copy_buffer_to_buffer(
            &self.index_stage_buffer,
            &self.index_buffer,
            Some(ind_bytes.len() as _),
        );
        self.vb_up_to_date = 1;
        Ok(())
    }
}

pub struct Renderer {
    tex_dset: rhi::DSet,
    render_outputs: Vec<rhi::RenderOutput>,
    render_depths: Vec<rhi::ImageView>,
    pfds: Vec<PerFrameData>,
    window: Arc<Window>,
    swapchain: rhi::Swapchain,
    sampler: rhi::Sampler,
    bg_image_view: rhi::ImageView,
    bg_image: rhi::Image,
    pipeline: rhi::RenderPipeline,
    camera: Cam3d,
    device: rhi::Device,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let device = rhi::Device::new(&window)?;
        let swapchain = device.create_swapchain()?;
        let bg_image = Self::load_bg_image(&device, "./default.png")?;
        let bg_image_view = bg_image.create_view(rhi::ViewDimension::D2, 0..1, 0..1)?;
        let sampler = device.create_sampler()?;
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
                    clear: true,
                    store: true,
                }],
                depth: Some(rhi::FragmentOutputInfo {
                    format: rhi::Format::D32Float,
                    clear: true,
                    store: true,
                }),
            },
            rhi::RasterMode::Fill(1.0),
            vec![
                vec![rhi::DBindingType::UBuffer(1)],
                vec![rhi::DBindingType::Sampler2d(1)],
            ],
            0,
        )?;
        let mut tex_dset = pipeline.new_set(1)?;
        tex_dset.write(vec![rhi::DBindingData::Sampler2d(vec![(
            &bg_image_view,
            &sampler,
        )])]);
        let mut camera = Cam3d {
            eye: glam::vec3(5.0, 5.0, 5.0),
            fov: 120.0,
            dir: glam::vec3(-1.0, -1.0, -1.0),
            aspect: 1.0,
            up: glam::vec3(0.0, 1.0, 0.0),
            padding: 0,
            proj_view: glam::Mat4::IDENTITY,
        };
        camera.update_proj_view();

        let pfds = (0..swapchain.images().len())
            .map(|i| PerFrameData::new(&device, &mut pipeline, i))
            .collect::<Result<_, _>>()?;

        let render_depths: Vec<_> = (0..swapchain.images().len())
            .map(|_| {
                let img = device.create_image(
                    rhi::Dimension::D2,
                    rhi::Format::D32Float,
                    swapchain.width(),
                    swapchain.height(),
                    1,
                    1,
                    rhi::ImageUsage::Attachment.into(),
                    rhi::MemLocation::Gpu,
                )?;
                img.create_view(rhi::ViewDimension::D2, 0..1, 0..1)
            })
            .collect::<Result<_, _>>()?;
        let render_outputs = (0..swapchain.images().len())
            .map(|i| pipeline.new_output(vec![&swapchain.views()[i], &render_depths[i]]))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            tex_dset,
            render_outputs,
            render_depths,
            pfds,
            window,
            device,
            swapchain,
            pipeline,
            camera,
            sampler,
            bg_image_view,
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
            rhi::ImageUsage::CopyDst | rhi::ImageUsage::CopySrc | rhi::ImageUsage::Sampled,
            rhi::MemLocation::Gpu,
        )?;
        let cmd_buffer = device.graphics_queue().create_command_buffer()?;
        let mut encoder = cmd_buffer.encoder()?;
        encoder.set_last_image_access(&image, rhi::ImageAccess::Undefined, 0..1, 0..1);
        encoder.copy_buffer_to_image(&stage_buffer, &image, 0..1, 0);
        encoder.set_last_image_access(
            &image,
            rhi::ImageAccess::Shader(rhi::RWAccess::Read),
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

    pub fn resize(
        &mut self,
        new_size: winit::dpi::PhysicalSize<u32>,
        redraw: bool,
    ) -> anyhow::Result<()> {
        for idx in 0..self.swapchain.images().len() {
            self.pfds[idx]
                .draw_sem
                .wait_for(self.pfds[idx].draw_sem_num, None)
                .inspect_err(|e| eprintln!("error waiting for running draws: {e}"))
                .ok();
        }
        self.render_outputs.clear();
        if let Err(e) = self.swapchain.resize(new_size.width, new_size.height) {
            eprintln!("resizing swapchain failed: {e}");
        }
        self.render_depths = (0..self.swapchain.images().len())
            .map(|_| {
                let img = self.device.create_image(
                    rhi::Dimension::D2,
                    rhi::Format::D32Float,
                    self.swapchain.width(),
                    self.swapchain.height(),
                    1,
                    1,
                    rhi::ImageUsage::Attachment.into(),
                    rhi::MemLocation::Gpu,
                )?;
                img.create_view(rhi::ViewDimension::D2, 0..1, 0..1)
            })
            .collect::<Result<_, _>>()?;
        self.render_outputs = (0..self.swapchain.images().len())
            .map(|i| {
                self.pipeline
                    .new_output(vec![&self.swapchain.views()[i], &self.render_depths[i]])
            })
            .collect::<Result<_, _>>()?;
        for pfd in &mut self.pfds {
            pfd.swapchain_image_initialized = 0;
        }
        if redraw {
            self.render()?;
        } else {
            self.window.request_redraw();
        }
        Ok(())
    }

    pub fn add_meshes(&mut self, meshes: Vec<Mesh>) {
        let meshes: Vec<_> = meshes.into_iter().map(Arc::new).collect();
        for pfd in &mut self.pfds {
            pfd.add_meshes(&meshes);
        }
    }

    pub fn clear_meshes(&mut self) {
        for pfd in &mut self.pfds {
            pfd.clear_meshes();
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let (mut img_idx, is_unoptimal) = self.swapchain.acquire_image()?;
        if is_unoptimal {
            let res = self.window.inner_size();
            self.resize(res, false)?;
            (img_idx, _) = self.swapchain.acquire_image()?;
        }
        let idx = img_idx as usize;
        self.pfds[idx]
            .draw_sem
            .wait_for(self.pfds[idx].draw_sem_num, None)?;
        if self.pfds[idx].swapchain_image_initialized == 1 {
            self.pfds[idx].swapchain_image_initialized = 2;
        }
        if self.pfds[idx].vb_up_to_date == 1 {
            self.pfds[idx].vb_up_to_date = 2;
        }
        let aspect_ratio = self.swapchain.images()[0].width() as f32
            / self.swapchain.images()[0].height().max(1) as f32;
        self.camera.aspect = aspect_ratio;
        self.camera.update_proj_view();
        self.pfds[idx]
            .camera_stage_buffer
            .write_data(bytemuck::bytes_of(&self.camera))?;
        let mut encoder = self.pfds[idx].cmd_buffer.encoder()?;
        encoder.copy_buffer_to_buffer(
            &self.pfds[idx].camera_stage_buffer,
            &self.pfds[idx].camera_buffer,
            None,
        );
        self.pfds[idx].update_vb(&mut encoder)?;
        if self.pfds[idx].swapchain_image_initialized == 0 {
            encoder.set_last_image_access_view(
                &self.swapchain.views()[idx],
                rhi::ImageAccess::Undefined,
            );
            encoder
                .set_last_image_access_view(&self.render_depths[idx], rhi::ImageAccess::Undefined);
            self.pfds[idx].swapchain_image_initialized = 1;
        } else {
            encoder.set_last_image_access_view(
                &self.swapchain.views()[idx],
                rhi::ImageAccess::Present,
            );
        }

        let mut render_pass = encoder.start_render_pipeline(
            &self.pipeline,
            &self.render_outputs[idx],
            vec![
                rhi::ClearValue::Colour([0.0; 4]),
                rhi::ClearValue::Depth(1.0, 0),
            ],
        );
        render_pass.bind_vbs(vec![&self.pfds[idx].vertex_buffer]);
        render_pass.bind_ib(&self.pfds[idx].index_buffer, rhi::IndexType::U16);
        render_pass.bind_dsets(vec![&self.pfds[idx].camera_dset, &self.tex_dset]);
        for draw_info in self.pfds[idx].mesh_offsets.values() {
            render_pass.draw_indexed(draw_info.vb_offset, draw_info.ib_offset, draw_info.len);
        }

        let mut encoder = render_pass.end();
        encoder.set_last_image_access_view(&self.swapchain.views()[idx], rhi::ImageAccess::Present);
        encoder.finalize()?;
        self.pfds[idx].draw_sem_num += 1;
        self.pfds[idx].cmd_buffer.submit(
            vec![],
            vec![
                self.pfds[idx]
                    .draw_sem
                    .submit_info(self.pfds[idx].draw_sem_num),
                self.pfds[idx].present_sem.submit_info(1),
            ],
        )?;

        self.window.pre_present_notify();
        self.swapchain
            .present_image(img_idx, &self.pfds[idx].present_sem)?;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        for idx in 0..self.swapchain.images().len() {
            self.pfds[idx]
                .draw_sem
                .wait_for(self.pfds[idx].draw_sem_num, None)
                .inspect_err(|e| eprintln!("error waiting for running draws: {e}"))
                .ok();
        }
        self.device.graphics_queue().wait_idle();
    }
}
