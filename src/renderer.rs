use std::sync::Arc;

use avk12::{
    MemoryLocation,
    ash::vk,
    canvas::{NextImageRes, PresentableImage},
    device::Device,
    pipeline::{
        AttachInfo, BindInfo, DSet, DSetWriteData, FragmentConfig, GraphicsPipeline,
        GraphicsPipelineCreateInfo, VertexAttribute, VertexConfig,
    },
    resource::{BufferCreateInfo, ImageAccess, ImageCreateInfo, ImageViewInfo},
    task::{ClearValue, DrawInfo},
};
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
    tex_dset: Arc<DSet>,
    transform: glam::Mat4,
    draw: bool,
}

pub struct Renderer {
    textures: HashMap<String, Arc<DSet>>,
    flat_sampler_dset: DSet,
    gp: GraphicsPipeline,
    device: Device,
}

impl Renderer {
    pub fn new(device: Device) -> anyhow::Result<Self> {
        let gp_info = GraphicsPipelineCreateInfo::builder()
            .set_layouts(vec![
                vec![BindInfo {
                    type_: vk::DescriptorType::SAMPLER,
                    count: 1,
                }],
                vec![BindInfo {
                    type_: vk::DescriptorType::SAMPLED_IMAGE,
                    count: 1,
                }],
            ])
            .vert_conf(
                VertexConfig::builder()
                    .shader("src/renderer/shaders/mesh.vert".to_string())
                    .attribs(vec![VertexAttribute::Vec4; 5])
                    .fn_name("main".to_string())
                    .stride(size_of::<Vertex>())
                    .build(),
            )
            .frag_conf(
                FragmentConfig::builder()
                    .shader("src/renderer/shaders/mesh.frag".to_string())
                    .fn_name("main".to_string())
                    .attachments(vec![AttachInfo {
                        format: device.canvas().info().surf_format().format,
                        clear: true,
                        store: true,
                    }])
                    .build(),
            )
            .build();
        let gp = device.new_graphics_pipeline(gp_info)?;
        let flat_sampler = device.new_sampler()?;
        let mut flat_sampler_dset = gp.new_set(0)?;
        flat_sampler_dset.update_binding(0, 0, DSetWriteData::Samplers(vec![&flat_sampler]))?;
        Ok(Self {
            textures: HashMap::new(),
            flat_sampler_dset,
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

    pub fn render(&mut self, camera: &Cam3d, mesh_draws: &[MeshDrawInfo]) -> anyhow::Result<()> {
        let swap_img = self.get_next_image(2)?;
        let swap_img_view = swap_img.image().view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            layer_range: 0..1,
            level_range: 0..1,
        })?;

        let mut recorder = self.device.new_task()?;
        // recorder.blit(&self.bg_image, swap_img.view().image().as_ref());
        let mut render_pass = recorder.graphics(
            &mut self.gp,
            vec![swap_img_view],
            vec![ClearValue::Color([0.4, 0.5, 0.3, 1.0])],
        )?;
        render_pass.bind_set(0, &self.flat_sampler_dset);
        for draw_info in mesh_draws {
            if draw_info.draw {
                render_pass.bind_vb(&draw_info.mesh_gpu.vert_buffer);
                render_pass.bind_ib(&draw_info.mesh_gpu.indx_buffer, true);
                render_pass.bind_set(1, &draw_info.tex_dset);
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

    pub fn load_texture(&mut self, path: &str) -> anyhow::Result<Arc<DSet>> {
        match self.textures.get(path).cloned() {
            Some(tex) => Ok(tex),
            None => {
                let img = image::open(path)?;
                let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
                let gpu_img = self.device.new_image(
                    ImageCreateInfo::builder()
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .res((img.width(), img.height(), 1))
                        .used_for(usage)
                        .build(),
                )?;
                let stage_buffer = self.device.new_buffer(
                    BufferCreateInfo::default()
                        .with_size(img.as_bytes().len() as _)
                        .with_used_for(vk::BufferUsageFlags::TRANSFER_SRC)
                        .with_mem_location(MemoryLocation::CpuToGpu),
                )?;
                stage_buffer.write_cpu(0, img.as_bytes())?;
                let mut cmd_rec = self.device.new_task()?;
                cmd_rec.copy_b2i(stage_buffer.slice(0..stage_buffer.len()), &gpu_img, 0, 0..1);
                cmd_rec.optimize_image_for(
                    &gpu_img,
                    ImageAccess {
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        access_flags: vk::AccessFlags::SHADER_READ,
                        access_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    },
                );
                cmd_rec.run()?.wait()?;
                let image_view = Arc::new(gpu_img.view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    layer_range: 0..1,
                    level_range: 0..1,
                })?);
                let mut tex_dset = self.gp.new_set(1)?;
                tex_dset.update_binding(0, 0, DSetWriteData::Textures(vec![&image_view]))?;
                let tex_dset = Arc::new(tex_dset);
                self.textures.insert(path.to_string(), tex_dset.clone());
                Ok(tex_dset)
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
        let tex_dset = self.load_texture(&tex_name)?;

        Ok(MeshDrawInfo {
            mesh_name,
            tex_name,
            mesh_gpu,
            tex_dset,
            transform: glam::Mat4::IDENTITY,
            draw,
        })
    }
}
