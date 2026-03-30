use std::sync::Arc;

use hashbrown::HashMap;
use rhi2::{
    Capped, HostAccess,
    buffer::{Buffer, BufferFlags},
    command::{CommandRecorder, GraphicsCommandRecorder},
    device::Device,
    graphics_pipeline::{
        AttachInfo, FragmentStageInfo, GraphicsPipeline, VertexAttribute, VertexStageInfo,
    },
    image::{Format, ImageFlags, ImageView},
    shader::{ShaderSetData, ShaderSetInfo},
    swapchain::{SCImageRes, Swapchain, SwapchainImage},
    sync::TaskFuture,
};

use crate::renderer::{
    camera::Cam3d,
    mesh::{GpuMesh, Mesh, Vertex},
    texture::Texture,
};

mod camera;
mod mesh;
mod texture;

pub struct PerFrameData<D: Device> {
    cam_stage_buffer: D::B,
    cam_buffer: Arc<D::B>,
    cam_sset: D::SS,
}

impl<D: Device> PerFrameData<D> {
    pub fn new(device: &D, gp: &mut D::GP) -> anyhow::Result<Self> {
        let cam_stage_buffer = device.new_buffer(
            size_of::<Cam3d>(),
            BufferFlags::CopySrc.into(),
            HostAccess::Write,
        )?;
        let cam_buffer = Arc::new(device.new_buffer(
            size_of::<Cam3d>(),
            BufferFlags::CopyDst | BufferFlags::Uniform,
            HostAccess::None,
        )?);
        let cam_sset = gp.new_set(
            0,
            vec![ShaderSetData::UniformBuffer(vec![Capped::Arc(
                cam_buffer.clone(),
            )])],
        )?;
        Ok(Self {
            cam_stage_buffer,
            cam_buffer,
            cam_sset,
        })
    }

    pub fn update_cam_cmds(&mut self, cam: &Cam3d, cr: &mut D::CR) -> anyhow::Result<()> {
        self.cam_stage_buffer
            .host_write(0, bytemuck::bytes_of(cam))?;
        cr.copy_b2b(
            &self.cam_stage_buffer,
            0,
            &self.cam_buffer,
            0,
            size_of::<Cam3d>(),
        );
        Ok(())
    }
}

pub struct Renderer<D: Device> {
    pfds: Vec<PerFrameData<D>>,
    cam: Cam3d,
    meshes: Vec<GpuMesh<D>>,
    textures: HashMap<String, Texture<D>>,
    gp: D::GP,
    bg_image: D::I,
    device: D,
}

impl<D: Device> Renderer<D> {
    pub fn new(device: D) -> anyhow::Result<Self> {
        let img = image::open("data/textures/alt/albedo.png")?;
        let bg_image = device.new_image(
            Format::Rgba8,
            (img.width(), img.height(), 1),
            1,
            ImageFlags::CopySrc | ImageFlags::CopyDst,
            HostAccess::None,
        )?;
        let mut bg_image_buffer = device.new_buffer(
            img.as_bytes().len(),
            BufferFlags::CopySrc.into(),
            HostAccess::Write,
        )?;
        bg_image_buffer.host_write(0, img.as_bytes())?;
        let mut cmd_rec = device.new_cmd_recorder()?;
        cmd_rec.copy_b2i(&bg_image_buffer, &bg_image);
        cmd_rec.run(vec![])?.wait()?;
        let mut gp = device.new_graphics_pipeline(
            "src/renderer2/shaders/triangle.wgsl",
            vec![vec![ShaderSetInfo::UniformBuffer(1)]],
            0,
            VertexStageInfo {
                entrypoint: "vs_main",
                attribs: vec![
                    VertexAttribute::Vec4,
                    VertexAttribute::Vec4,
                    VertexAttribute::Vec4,
                    VertexAttribute::Vec4,
                    VertexAttribute::Vec4,
                ],
                stride: size_of::<Vertex>(),
            },
            FragmentStageInfo {
                entrypoint: "fs_main",
                outputs: vec![AttachInfo {
                    format: device.swapchain().fmt(),
                    clear: false,
                    store: true,
                }],
                depth: None,
            },
        )?;
        let cam = Cam3d::new(
            glam::Vec3 {
                x: 3.0,
                y: 3.0,
                z: 3.0,
            },
            glam::Vec3 {
                x: -1.0,
                y: -1.0,
                z: -1.0,
            },
            glam::Vec3::Y,
            3.0,
            1.0,
        );
        let mut pfds = vec![];
        for _ in 0..device.swapchain().img_count() {
            let pfd = PerFrameData::new(&device, &mut gp)?;
            pfds.push(pfd);
        }
        Ok(Self {
            pfds,
            cam,
            meshes: vec![],
            textures: HashMap::new(),
            gp,
            bg_image,
            device,
        })
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        self.device.swapchain_mut().refresh_res()?;
        Ok(())
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let mut swap_img = match self.device.swapchain_mut().next_image() {
            SCImageRes::Success(swap) => swap,
            SCImageRes::Unavailable => return Ok(()),
            SCImageRes::Outdated => return Ok(()),
            SCImageRes::Error(e) => return Err(anyhow::Error::msg(e)),
        };

        let mut recorder = self.device.new_cmd_recorder()?;
        recorder.blit(&self.bg_image, swap_img.view().image().as_ref());
        let mut render_pass = recorder
            .graphics(
                &mut self.gp,
                vec![&swap_img.view()],
                vec![[0.5; 4]],
                None,
                None,
            )
            .map_err(|(e, _)| e)?;
        let recorder = D::CR::finish_graphics(render_pass);
        let mut draw_tf = recorder.run(vec![])?;
        draw_tf.wait()?;
        swap_img.present()?;

        Ok(())
    }
}

impl<D: Device> Drop for Renderer<D> {
    fn drop(&mut self) {
        self.device.wait_idle();
    }
}
