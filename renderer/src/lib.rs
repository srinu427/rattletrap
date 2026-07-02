use std::sync::Arc;

use anyhow::Context;
use avk12::{
    MemoryLocation,
    ash::vk,
    canvas::{NextImageRes, PresentableImage},
    device::Device,
    pipeline::{DSet, DSetWriteData},
    resource::{
        BufferCreateInfo, BufferRef, ImageAccess, ImageCreateInfo, ImageRef, ImageViewInfo,
    },
    task::{ClearValue, DrawInfo},
};
use common::Entity;
use getset::Setters;
use hashbrown::HashMap;
use indexmap::IndexMap;

use crate::{
    camera::Cam3d,
    mesh::{GpuMesh, Mesh, MeshCreateInfo},
    mesh_pipeline::MeshPipeline,
};

pub mod camera;
pub mod mesh;
pub mod mesh_pipeline;
pub mod texture;

#[derive(Setters)]
pub struct MeshDrawInfo {
    mesh_name: String,
    tex_name: String,
    mesh_gpu: GpuMesh,
    tex_dset: Arc<DSet>,
    #[getset(set = "pub")]
    transform: glam::Mat4,
    #[getset(set = "pub")]
    draw: bool,
}

pub struct PerFrameData {
    camera_stage_buffer: BufferRef,
    camera_buffer: BufferRef,
    camera_dset: DSet,
    mesh_transform_buffer: BufferRef,
    mesh_transform_stage_buffer: BufferRef,
    mesh_transform_dset: DSet,
    depth_image: ImageRef,
}

impl PerFrameData {
    pub fn new(device: &Device, mesh_pipeline: &MeshPipeline) -> anyhow::Result<Self> {
        let camera_stage_buffer = device
            .new_buffer(BufferCreateInfo {
                size: size_of::<Cam3d>() as _,
                used_for: vk::BufferUsageFlags::TRANSFER_SRC,
                mem_location: MemoryLocation::CpuToGpu,
            })
            .context("camera stage buffer creation failed")?;
        let camera_buffer = device
            .new_buffer(BufferCreateInfo {
                size: size_of::<Cam3d>() as _,
                used_for: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
                mem_location: MemoryLocation::GpuOnly,
            })
            .context("camera buffer creation failed")?;
        let mut camera_dset = mesh_pipeline
            .new_cam_dset()
            .context("camera descriptor set creation failed")?;
        camera_dset
            .update_binding(0, 0, DSetWriteData::Buffers(vec![&camera_buffer]))
            .context("updating camera descriptor set data failed")?;
        let mesh_transform_stage_buffer = device
            .new_buffer(BufferCreateInfo {
                size: 128 * size_of::<glam::Mat4>() as u64,
                used_for: vk::BufferUsageFlags::TRANSFER_SRC,
                mem_location: MemoryLocation::CpuToGpu,
            })
            .context("mesh transform stage buffer creation failed")?;
        let mesh_transform_buffer = device
            .new_buffer(BufferCreateInfo {
                size: 128 * size_of::<glam::Mat4>() as u64,
                used_for: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                mem_location: MemoryLocation::GpuOnly,
            })
            .context("mesh transform buffer creation failed")?;
        let mut mesh_transform_dset = mesh_pipeline
            .new_model_transforms_dset()
            .context("mesh transform descriptor set creation failed")?;
        mesh_transform_dset.update_binding(
            0,
            0,
            DSetWriteData::Buffers(vec![&mesh_transform_buffer]),
        )?;
        let depth_image = device
            .new_image(
                ImageCreateInfo::builder()
                    .format(vk::Format::D24_UNORM_S8_UINT)
                    .res((1, 1, 1))
                    .used_for(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .build(),
            )
            .context("depth image creation failed")?;
        Ok(Self {
            camera_dset,
            camera_stage_buffer,
            camera_buffer,
            mesh_transform_stage_buffer,
            mesh_transform_buffer,
            mesh_transform_dset,
            depth_image,
        })
    }
}

pub struct Renderer {
    per_frame_datas: Vec<PerFrameData>,
    textures: HashMap<String, Arc<DSet>>,
    flat_sampler_dset: DSet,
    gp: MeshPipeline,
    device: Device,
}

impl Renderer {
    pub fn new(device: Device) -> anyhow::Result<Self> {
        let gp = MeshPipeline::new(&device)?;
        let flat_sampler = device.new_sampler()?;
        let mut flat_sampler_dset = gp.new_sampler_dset()?;
        flat_sampler_dset.update_binding(0, 0, DSetWriteData::Samplers(vec![&flat_sampler]))?;
        let per_frame_datas: Vec<_> = (0..device.canvas().image_count())
            .map(|_| PerFrameData::new(&device, &gp))
            .collect::<Result<_, _>>()
            .context("frame specific data creation failed")?;
        Ok(Self {
            per_frame_datas,
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

    pub fn render(
        &mut self,
        camera: &mut Cam3d,
        mesh_draws: &IndexMap<Entity, MeshDrawInfo>,
    ) -> anyhow::Result<()> {
        let swap_img = self.get_next_image(2)?;
        let swap_img_view = swap_img.image().view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            layer_range: 0..1,
            level_range: 0..1,
        })?;

        let frame_idx = swap_img.idx() as usize;

        let canvas_res = self.device.canvas().info().res();
        let aspect = canvas_res.0 as f32 / canvas_res.1.max(1) as f32;
        camera.set_aspect(aspect);
        camera.update_proj_view();
        self.per_frame_datas[frame_idx]
            .camera_stage_buffer
            .write_cpu(0, bytemuck::bytes_of(camera))
            .context("writing camera info to buffer failed")?;
        let depth_res = self.per_frame_datas[frame_idx].depth_image.res();
        if canvas_res != (depth_res.0, depth_res.1) {
            self.per_frame_datas[frame_idx].depth_image = self
                .device
                .new_image(
                    ImageCreateInfo::builder()
                        .format(vk::Format::D24_UNORM_S8_UINT)
                        .res((canvas_res.0, canvas_res.1, 1))
                        .used_for(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                        .build(),
                )
                .context("depth image resize failed")?;
        }
        let depth_image_view =
            self.per_frame_datas[frame_idx]
                .depth_image
                .view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    layer_range: 0..1,
                    level_range: 0..1,
                })?;
        // Update object transforms
        let obj_transforms: Vec<_> = mesh_draws.values().map(|v| v.transform).collect();
        let transforms_len = obj_transforms.len() * size_of::<glam::Mat4>();
        let req_buffer_size = transforms_len.next_power_of_two() as u64;
        if self.per_frame_datas[frame_idx]
            .mesh_transform_stage_buffer
            .len()
            < req_buffer_size
        {
            self.per_frame_datas[frame_idx].mesh_transform_stage_buffer = self
                .device
                .new_buffer(BufferCreateInfo {
                    size: req_buffer_size,
                    used_for: vk::BufferUsageFlags::TRANSFER_SRC,
                    mem_location: MemoryLocation::CpuToGpu,
                })
                .context("mesh transform stage buffer creation failed")?;
        }
        if self.per_frame_datas[frame_idx].mesh_transform_buffer.len() < req_buffer_size {
            self.per_frame_datas[frame_idx].mesh_transform_buffer = self
                .device
                .new_buffer(BufferCreateInfo {
                    size: req_buffer_size,
                    used_for: vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    mem_location: MemoryLocation::GpuOnly,
                })
                .context("mesh transform stage buffer creation failed")?;
        }
        self.per_frame_datas[frame_idx]
            .mesh_transform_stage_buffer
            .write_cpu(0, bytemuck::cast_slice(&obj_transforms))?;

        let mut recorder = self.device.new_task()?;
        // Copy camera info to gpu
        recorder.copy_b2b(
            self.per_frame_datas[frame_idx]
                .camera_stage_buffer
                .full_slice(),
            self.per_frame_datas[frame_idx].camera_buffer.full_slice(),
        );
        recorder.copy_b2b(
            self.per_frame_datas[frame_idx]
                .mesh_transform_stage_buffer
                .slice(0..transforms_len as u64),
            self.per_frame_datas[frame_idx]
                .mesh_transform_buffer
                .slice(0..transforms_len as u64),
        );
        // recorder.blit(&self.bg_image, swap_img.view().image().as_ref());
        let mut render_pass = recorder.graphics(
            &mut self.gp.gp,
            vec![swap_img_view, depth_image_view],
            vec![
                ClearValue::Color([0.4, 0.5, 0.3, 1.0]),
                ClearValue::Depth(1.0),
            ],
        )?;
        render_pass.bind_set(0, &self.per_frame_datas[frame_idx].camera_dset);
        render_pass.bind_set(1, &self.per_frame_datas[frame_idx].mesh_transform_dset);
        render_pass.bind_set(2, &self.flat_sampler_dset);
        for (obj_id, (_, draw_info)) in mesh_draws.iter().enumerate() {
            if draw_info.draw {
                render_pass.bind_vb(&draw_info.mesh_gpu.vert_buffer);
                render_pass.bind_ib(&draw_info.mesh_gpu.indx_buffer, true);
                render_pass.bind_set(3, &draw_info.tex_dset);
                render_pass.set_pc(&(obj_id as u32).to_le_bytes());
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
                cmd_rec.copy_b2i(stage_buffer.full_slice(), &gpu_img, 0, 0..1);
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
                let mut tex_dset = self.gp.new_texture_dset()?;
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
