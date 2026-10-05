use std::sync::Arc;

use ash::vk;
use bytemuck::NoUninit;
use ecs::EcsData;
use glam::Vec4Swizzles;
use hashbrown::HashMap;
use indexmap::IndexMap;
use naga::back::spv;
use naga::front::glsl;
use naga::valid;
use winit::window::Window;

mod camera_descriptor;
mod mesh_pipeline;
mod scene_descriptor;
mod smap_pipeline;

use crate::helpers::buffer::GpuVec;
use crate::helpers::device::{GpuCommandRecorder, GpuCtx};
use crate::helpers::image::{GpuDsl, GpuImage, GpuImageAccess};
use crate::helpers::swapchain::GpuSwapchain;
use crate::renderer::camera_descriptor::CameraDescriptor;
use crate::renderer::mesh_pipeline::MeshPipeline;
use crate::renderer::scene_descriptor::SceneDescriptor;
use crate::{Camera3d, DrawableMesh, Light, Material, Mesh, MeshVertex, load_ron};

fn convert_glsl_to_spv(glsl_source: &str, stage: naga::ShaderStage) -> anyhow::Result<Vec<u32>> {
    let options = glsl::Options {
        stage,
        defines: Default::default(),
    };

    let mut frontend = glsl::Frontend::default();
    let module = frontend.parse(&options, glsl_source)?;

    let mut validator =
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all());
    let module_info = validator.validate(&module)?;

    let writer_flags = spv::WriterFlags::empty();
    let mut writer_options = spv::Options::default();
    writer_options.flags = writer_flags;

    let mut writer = spv::Writer::new(&writer_options)?;
    let mut spv_words = vec![];
    writer.write(&module, &module_info, None, &None, &mut spv_words)?;

    Ok(spv_words)
}

fn load_glsl(
    ctx: &GpuCtx,
    code: &str,
    stage: naga::ShaderStage,
) -> anyhow::Result<vk::ShaderModule> {
    let spv_words = convert_glsl_to_spv(code, stage)?;
    let shader_mod = unsafe {
        ctx.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&spv_words),
            None,
        )?
    };
    Ok(shader_mod)
}

struct GpuLoadedMesh {
    vbo: GpuVec<MeshVertex>,
    ibo: GpuVec<u16>,
    draw_count: u32,
}

impl GpuLoadedMesh {
    fn new(ctx: &mut GpuCtx, cr: &mut GpuCommandRecorder, mesh: Mesh) -> anyhow::Result<Self> {
        let mut vbo = GpuVec::with_capacity(
            ctx,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            mesh.vertices.len(),
        )?;
        let mut ibo = GpuVec::with_capacity(
            ctx,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            mesh.indices.len(),
        )?;
        vbo.write(ctx, 0, &mesh.vertices, cr)?;
        ibo.write(ctx, 0, &mesh.indices, cr)?;
        let draw_count = mesh.indices.len() as u32;
        Ok(Self {
            vbo,
            ibo,
            draw_count,
        })
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        self.vbo.destroy(ctx);
        self.ibo.destroy(ctx);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct GpuMeshInfo {
    transform: glam::Mat4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct GpuCamera {
    transform: glam::Mat4,
    eye: glam::Vec4,
}

impl GpuCamera {
    fn from_cam_3d(cam3d: &Camera3d) -> Self {
        Self {
            transform: cam3d.get_perspective_proj(),
            eye: glam::Vec4::from((cam3d.eye, 1.0)),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit, bytemuck::Zeroable)]
struct GpuLightData {
    transform: glam::Mat4, // Updated once every frame
    position: glam::Vec4,  // xyz = World Position, w = Type (0 = Directional, 1 = Point)
    color: glam::Vec4,     // rgb = Light Color, a = Intensity / Power
    direction: glam::Vec4, // xyz = Light Direction (for directional lights), w = Attenuation distance (for point lights)
}

impl GpuLightData {
    fn from_light(light: &Light) -> Self {
        let mut out = Self {
            transform: glam::Mat4::IDENTITY,
            position: light.position,
            color: light.color,
            direction: light.direction,
        };
        out.refresh_transform();
        out
    }

    fn refresh_transform(&mut self) {
        let ycomp = self.direction.xyz().dot(glam::Vec3::Y);
        let up = if ycomp.abs() > 99.9 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let view = glam::Mat4::look_to_rh(self.position.xyz(), self.direction.xyz(), up);
        let proj = if self.position.z == 0.0 {
            glam::Mat4::orthographic_rh(100.0, 100.0, 100.0, 100.0, 0.1, 100.0)
        } else {
            glam::Mat4::perspective_rh(90.0, 1.0, 0.1, 100.0)
        };
        self.transform = proj * view;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct DrawableIds {
    mesh_buffer_idx: u32,
    mesh_info_idx: u32,
    material_idx: u32,
}

pub struct RendererVk12 {
    deferred_command_buffer: Option<GpuCommandRecorder>,
    mesh_buffers: Vec<GpuLoadedMesh>,
    scene_desc: SceneDescriptor,
    camera_desc: CameraDescriptor,
    depth_image: GpuImage,
    mesh_pipeline: MeshPipeline,
    scene_dsl: GpuDsl,
    camera_dsl: GpuDsl,
    swapchain: GpuSwapchain,
    ctx: GpuCtx,
}

impl RendererVk12 {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let mut ctx = GpuCtx::new(window)?;
        let swapchain = GpuSwapchain::new(&mut ctx)?;
        let mut camera_dsl = GpuDsl::new(&mut ctx, vec![(vk::DescriptorType::UNIFORM_BUFFER, 1)])?;
        let mut scene_dsl = GpuDsl::new(
            &mut ctx,
            vec![
                (vk::DescriptorType::STORAGE_BUFFER, 1),
                (vk::DescriptorType::STORAGE_BUFFER, 1),
                (vk::DescriptorType::STORAGE_BUFFER, 1),
            ],
        )?;
        let mesh_pipeline = MeshPipeline::new(
            &mut ctx,
            swapchain.format.format,
            &mut camera_dsl,
            &mut scene_dsl,
        )?;
        let scene_desc = SceneDescriptor::new(&mut ctx, &mut scene_dsl)?;
        let camera_desc = CameraDescriptor::new(&mut ctx, &mut camera_dsl)?;
        let depth_image = GpuImage::new(
            &mut ctx,
            vk::ImageType::TYPE_2D,
            vk::Format::D32_SFLOAT,
            (swapchain.res.0, swapchain.res.1, 1),
            1,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        Ok(Self {
            deferred_command_buffer: None,
            mesh_buffers: Default::default(),
            scene_desc,
            camera_desc,
            depth_image,
            mesh_pipeline,
            camera_dsl,
            scene_dsl,
            swapchain,
            ctx,
        })
    }

    fn get_deferred_cmd_buffer(&mut self) -> anyhow::Result<GpuCommandRecorder> {
        let cr = self.deferred_command_buffer.take();
        let cr = match cr {
            Some(t) => t,
            None => {
                let cr = self.ctx.get_command_recorder()?;
                cr
            }
        };
        Ok(cr)
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        self.swapchain.resize(&mut self.ctx)?;
        Ok(())
    }

    pub fn reload_resources(&mut self, ecs_data: &mut EcsData) -> anyhow::Result<()> {
        ecs_data.remove_component_all_entities::<DrawableIds>();
        let mut cr = self.get_deferred_cmd_buffer()?;
        for mut mesh in self.mesh_buffers.drain(..) {
            mesh.destroy(&mut self.ctx);
        }
        let mut new_meshes = IndexMap::new();
        let mut new_materials = IndexMap::new();
        let mut obj_idxs = Vec::new();
        if let Some(dm_it) = ecs_data.comp_data_iter::<DrawableMesh>() {
            for (i, (ent, drawable)) in dm_it.enumerate() {
                let mesh_idx = match new_meshes.get_index_of(&drawable.mesh) {
                    Some(t) => t,
                    None => {
                        let mesh: Mesh = match load_ron(&drawable.mesh) {
                            Ok(t) => t,
                            Err(e) => {
                                log::error!("loading mesh {} failed: {e}", &drawable.mesh);
                                continue;
                            }
                        };
                        new_meshes.insert_full(drawable.mesh.clone(), mesh).0
                    }
                };

                let material_idx = match new_materials.get_index_of(&drawable.material) {
                    Some(t) => t,
                    None => {
                        let material: Material = load_ron(&drawable.material).unwrap_or_default();
                        new_materials
                            .insert_full(drawable.material.clone(), material)
                            .0
                    }
                };

                let drawable_idxs = DrawableIds {
                    mesh_buffer_idx: mesh_idx as _,
                    mesh_info_idx: i as _,
                    material_idx: material_idx as _,
                };

                obj_idxs.push((ent, drawable_idxs));
            }
        }
        // Add the DrawableIds components
        for (ent, di) in obj_idxs {
            ecs_data.insert_component(ent, di);
        }

        let mut lights = vec![];
        if let Some(l_it) = ecs_data.comp_data_iter::<Light>() {
            for (_, l) in l_it {
                lights.push(GpuLightData::from_light(l));
            }
        }
        let lights: Vec<_> = ecs_data
            .comp_data_iter::<Light>()
            .map(|x| x.map(|(_, l)| GpuLightData::from_light(l)).collect())
            .unwrap_or_default();

        let mut new_mesh_buffers = vec![];
        for (_, mesh) in new_meshes {
            let new_mesh_buffer = GpuLoadedMesh::new(&mut self.ctx, &mut cr, mesh)?;
            new_mesh_buffers.push(new_mesh_buffer);
        }
        let new_materials: Vec<_> = new_materials.into_values().collect();

        self.mesh_buffers = new_mesh_buffers;
        self.scene_desc.mesh_infos.clear();
        self.scene_desc.materials.clear();
        self.scene_desc.lights.clear();
        self.scene_desc
            .udpate_materials(&mut self.ctx, &mut cr, &new_materials)?;
        self.scene_desc
            .update_lights(&mut self.ctx, &mut cr, &lights)?;

        self.deferred_command_buffer = Some(cr);
        Ok(())
    }

    pub fn render(&mut self, ecs_data: &mut EcsData, camera: &Camera3d) -> anyhow::Result<()> {
        let Some(idx) = self.swapchain.acquire(&mut self.ctx)? else {
            self.resize()?;
            return Ok(());
        };
        let drawables: HashMap<_, _> = ecs_data
            .comp_data_iter::<DrawableMesh>()
            .map(|dm_it| dm_it.map(|dm| (dm.0, dm.1.clone())).collect())
            .unwrap_or_default();

        let mut object_datas = vec![
            GpuMeshInfo {
                transform: glam::Mat4::IDENTITY
            };
            drawables.len()
        ];

        let drawable_idxs: Vec<_> = ecs_data
            .comp_data_iter::<DrawableIds>()
            .map(|di_it| di_it.map(|di| (di.0, di.1.clone())).collect())
            .unwrap_or_default();

        for (ent, di) in drawable_idxs.iter() {
            if let Some(dm) = drawables.get(ent) {
                object_datas[di.mesh_info_idx as usize].transform = dm.transform;
            }
        }
        let drawable_idxs: Vec<_> = drawable_idxs.into_iter().map(|x| x.1).collect();

        let mut cr = self.get_deferred_cmd_buffer()?;

        self.camera_desc
            .update_camera(&mut self.ctx, &mut cr, &GpuCamera::from_cam_3d(camera))?;
        self.scene_desc
            .udpate_mesh_infos(&mut self.ctx, &mut cr, &object_datas)?;

        let sw_res = self.swapchain.images[idx as usize].res;
        if self.depth_image.res != sw_res {
            let depth_image = GpuImage::new(
                &mut self.ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::D32_SFLOAT,
                (self.swapchain.res.0, self.swapchain.res.1, 1),
                1,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )?;
            self.depth_image.destroy(&mut self.ctx);
            self.depth_image = depth_image;
        }
        self.swapchain.images[idx as usize].transition(
            &self.ctx,
            cr.cb,
            GpuImageAccess {
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
        );
        self.depth_image.transition(
            &self.ctx,
            cr.cb,
            GpuImageAccess {
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            },
        );

        // println!(
        //     "{}:{}:{}",
        //     self.scene_desc.lights.len,
        //     self.scene_desc.materials.len,
        //     self.scene_desc.mesh_infos.len
        // );

        self.mesh_pipeline.draw_meshes(
            &mut self.ctx,
            &mut cr,
            &self.camera_desc,
            &self.scene_desc,
            &self.mesh_buffers,
            &drawable_idxs,
            &mut self.swapchain.images[idx as usize],
            &mut self.depth_image,
        )?;

        self.swapchain.images[idx as usize].transition(
            &mut self.ctx,
            cr.cb,
            GpuImageAccess {
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                access: vk::AccessFlags::empty(),
                stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            },
        );

        let task = cr.submit(&mut self.ctx)?;
        task.wait(&mut self.ctx)?;
        self.deferred_command_buffer = None;
        self.swapchain.present(&mut self.ctx, idx)?;
        Ok(())
    }
}

impl Drop for RendererVk12 {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.ctx.device.device_wait_idle() {
                log::warn!("waiting for gpu to be idle failed: {e}")
            };
            if let Some(mut dcr) = self.deferred_command_buffer.take() {
                dcr.destroy(&mut self.ctx);
            }
            for mut gmesh in self.mesh_buffers.drain(..) {
                gmesh.destroy(&mut self.ctx);
            }
            self.camera_desc
                .destroy(&mut self.ctx, &mut self.camera_dsl);
            self.camera_dsl.destroy(&mut self.ctx);
            self.scene_desc.destroy(&mut self.ctx, &mut self.scene_dsl);
            self.scene_dsl.destroy(&mut self.ctx);
            self.mesh_pipeline.destroy(&mut self.ctx);
            self.depth_image.destroy(&mut self.ctx);
            self.swapchain.destroy(&mut self.ctx);
        }
    }
}
