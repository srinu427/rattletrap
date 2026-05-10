use std::sync::Arc;

use anyhow::Ok;
use avk12::{
    canvas::{NextImageRes, PresentableImage},
    device::Device,
    pipeline::{
        AttachInfo, BindInfo, DSet, DSetWriteData, FragmentConfig, GraphicsPipeline,
        GraphicsPipelineCreateInfo, RasterConfig, RasterMode, VertexAttribute, VertexConfig,
    },
    resource::{
        Buffer, BufferCreateInfo, BufferUsageFlag, DataMoveDir, Format, Image, ImageCreateInfo,
        ImageUsageFlag, ImageView, ImageViewInfo, ImageViewType,
    },
    task::{ClearValue, DrawInfo, Task},
};
use enumflags2::BitFlags;
use hashbrown::HashMap;

use crate::renderer::{
    camera::Cam3d,
    mesh::{GpuMesh, Mesh, MeshCreateInfo, Vertex},
};

pub mod camera;
pub mod mesh;
pub mod texture;

pub struct MeshDrawInfo {
    mesh_name: String,
    tex_name: String,
    mesh_gpu: GpuMesh,
    tex_view: Arc<ImageView>,
    transform: glam::Mat4,
    draw: bool,
}

pub struct Renderer {
    textures: HashMap<String, Arc<ImageView>>,
    gp: GraphicsPipeline,
    device: Device,
}

impl Renderer {
    fn load_plain_image(
        device: &mut Device,
        path: &str,
        usage: BitFlags<ImageUsageFlag>,
    ) -> anyhow::Result<Image> {
        let img = image::open(path)?;
        let usage = usage | ImageUsageFlag::CopyDst;
        let gpu_img = device.new_image(
            ImageCreateInfo::builder()
                .format(Format::Rgba8)
                .res((img.width(), img.height(), 1))
                .used_for(usage)
                .build(),
        )?;
        let stage_buffer = device.new_buffer(
            BufferCreateInfo::builder()
                .size(img.as_bytes().len() as _)
                .used_for(BufferUsageFlag::CopySrc.into())
                .data_move_dir(DataMoveDir::Cpu2Gpu)
                .build(),
        )?;
        stage_buffer.write_cpu(0, img.as_bytes())?;
        let mut cmd_rec = device.new_task()?;
        cmd_rec.copy_b2i(stage_buffer.view(0..stage_buffer.len()), &gpu_img, 0, 0..1);
        cmd_rec.run()?.wait()?;
        Ok(gpu_img)
    }

    pub fn new(mut device: Device) -> anyhow::Result<Self> {
        let gp_info = GraphicsPipelineCreateInfo::builder()
            .shader("src/renderer/shaders/mesh.wgsl".to_string())
            .set_layouts(vec![])
            .vert_conf(
                VertexConfig::builder()
                    .attribs(vec![VertexAttribute::Vec4; 5])
                    .fn_name("vs_main".to_string())
                    .stride(size_of::<Vertex>())
                    .build(),
            )
            .frag_conf(
                FragmentConfig::builder()
                    .fn_name("fs_main".to_string())
                    .attachments(vec![AttachInfo {
                        format: device.canvas().info().format(),
                        clear: true,
                        store: true,
                    }])
                    .build(),
            )
            .build();
        let gp = device.new_graphics_pipeline(gp_info)?;
        Ok(Self {
            textures: HashMap::new(),
            gp,
            device,
        })
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        self.device.canvas_mut().refresh_res()?;
        Ok(())
    }

    fn get_next_image(&mut self, retries: usize) -> anyhow::Result<PresentableImage> {
        match self.device.canvas_mut().get_next_image() {
            NextImageRes::Success(image) => Ok(image),
            NextImageRes::NeedCanvasRefresh => {
                if retries == 0 {
                    return Err(anyhow::Error::msg("canvas needs refresh"));
                }
                self.resize()?;
                self.get_next_image(retries - 1)
            }
            NextImageRes::Error(e) => Err(anyhow::Error::msg(e)),
        }
    }

    pub fn render(&mut self, mesh_draws: &[MeshDrawInfo]) -> anyhow::Result<()> {
        let swap_img = self.get_next_image(2)?;
        let swap_img_view = swap_img.image().view(&ImageViewInfo {
            view_type: ImageViewType::E2d,
            layer_range: 0..1,
            level_range: 0..1,
        })?;

        let mut recorder = self.device.new_task()?;
        // recorder.blit(&self.bg_image, swap_img.view().image().as_ref());
        let mut render_pass = recorder.graphics(
            &mut self.gp,
            vec![swap_img_view],
            vec![ClearValue::Color([1.0, 0.0, 0.0, 1.0])],
        )?;
        for draw_info in mesh_draws {
            if draw_info.draw {
                render_pass.bind_vb(&draw_info.mesh_gpu.vert_buffer);
                render_pass.bind_ib(&draw_info.mesh_gpu.indx_buffer, true);
                render_pass.draw(vec![DrawInfo::Indexed {
                    vb_offset: 0,
                    ib_offset: 0,
                    count: draw_info.mesh_gpu.indx_count,
                }]);
            }
        }
        drop(render_pass);
        recorder.run()?.wait()?;
        swap_img.present()?;
        Ok(())
    }

    pub fn load_texture(&mut self, path: &str) -> anyhow::Result<Arc<ImageView>> {
        match self.textures.get(path).cloned() {
            Some(tex) => Ok(tex),
            None => {
                let image =
                    Self::load_plain_image(&mut self.device, path, ImageUsageFlag::Sampled.into())?;
                let image_view = Arc::new(image.view(&ImageViewInfo {
                    view_type: ImageViewType::E2d,
                    layer_range: 0..1,
                    level_range: 0..1,
                })?);
                self.textures.insert(path.to_string(), image_view.clone());
                Ok(image_view)
            }
        }
    }

    pub fn load_mesh_draw(
        &mut self,
        mesh_name: String,
        mesh_create_info: MeshCreateInfo,
        tex_name: String,
        draw: bool,
    ) -> anyhow::Result<MeshDrawInfo> {
        let mesh_cpu = Mesh::new(mesh_create_info)?;
        let mesh_gpu = GpuMesh::new(&self.device, &mesh_cpu)?;
        let tex_view = self.load_texture(&tex_name)?;
        Ok(MeshDrawInfo {
            mesh_name,
            tex_name,
            mesh_gpu,
            tex_view,
            transform: glam::Mat4::IDENTITY,
            draw,
        })
    }
}
