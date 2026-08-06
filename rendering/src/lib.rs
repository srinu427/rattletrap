use std::sync::Arc;

use anyhow::Context;
use ash::vk;
use common::Entity;
use gpu_allocator::MemoryLocation;
use indexmap::IndexMap;
use winit::window::Window;

use crate::{
    camera::Camera,
    tex_mesh::{GpuMesh, Mesh, TexMeshPass},
    vkraii::{
        command::CommandBufferRaii,
        device::DeviceRaii,
        resource::{BufferRaii, ImageAccess, ImageRaii, ImageViewKey},
        swapchain::SwapchainRaii,
    },
};

pub mod camera;
pub mod tex_mesh;
mod vkraii;

pub struct PerFrameData {
    pub depth_image: ImageRaii,
}

impl PerFrameData {
    pub fn new(device: &DeviceRaii, res: (u32, u32)) -> anyhow::Result<Self> {
        let depth_image = ImageRaii::new(
            &device.device_d,
            &device.allocator,
            &vk::ImageCreateInfo::default()
                .array_layers(1)
                .extent(vk::Extent3D {
                    width: res.0,
                    height: res.1,
                    depth: 1,
                })
                .format(vk::Format::D32_SFLOAT)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .mip_levels(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            MemoryLocation::GpuOnly,
        )?;
        Ok(Self { depth_image })
    }
}

pub struct RenderingManager {
    pfd_idx: usize,
    pub per_frame_datas: Vec<PerFrameData>,
    pub meshes: IndexMap<Entity, GpuMesh>,
    pub camera: Camera,
    textures: IndexMap<String, ImageRaii>,
    pipeline: TexMeshPass,
    deferred_cb: Option<CommandBufferRaii>,
    swapchain: SwapchainRaii,
    device: DeviceRaii,
}

impl RenderingManager {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let mut device = DeviceRaii::new(window)?;
        let swapchain = SwapchainRaii::new(&device.device_d)?;
        let pipeline = TexMeshPass::new(&mut device, swapchain.format)?;
        let camera = Camera {
            eye: glam::vec3(0.0, 0.0, 2.0),
            dir: -glam::Vec3::Z,
            up: glam::Vec3::Y,
            fov: 1.5,
            aspect: 1.0,
        };
        let per_frame_datas: Vec<_> = (0..swapchain.images.len())
            .map(|_| PerFrameData::new(&device, swapchain.res))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            pfd_idx: 0,
            per_frame_datas,
            meshes: Default::default(),
            camera,
            textures: Default::default(),
            pipeline,
            deferred_cb: None,
            swapchain,
            device,
        })
    }

    pub fn refresh_size(&mut self) -> anyhow::Result<()> {
        self.swapchain.refresh()?;
        self.camera.aspect =
            self.swapchain.res.0.max(1) as f32 / self.swapchain.res.1.max(1) as f32;
        let new_pfds: Vec<_> = (0..self.swapchain.images.len())
            .map(|_| PerFrameData::new(&self.device, self.swapchain.res))
            .collect::<Result<_, _>>()?;
        self.per_frame_datas = new_pfds;
        self.pfd_idx = 0;
        Ok(())
    }

    fn get_deferred_cb(&mut self) -> anyhow::Result<CommandBufferRaii> {
        match self.deferred_cb.take() {
            Some(t) => Ok(t),
            None => {
                let mut cb = self.device.command_pool.get_cb()?;
                cb.begin()?;
                Ok(cb)
            }
        }
    }

    pub fn load_mesh(&mut self, mesh: Mesh) -> anyhow::Result<GpuMesh> {
        let mut deferred_cb = self.get_deferred_cb()?;
        let gpu_mesh = GpuMesh::new(&mut self.device, &mut deferred_cb, &mut self.pipeline, mesh)?;
        self.deferred_cb = Some(deferred_cb);
        Ok(gpu_mesh)
    }

    fn load_image(&mut self, path: &str) -> anyhow::Result<()> {
        if self.textures.contains_key(path) {
            return Ok(());
        }
        let img_obj = image::open(path)?;
        let img_bytes = img_obj.to_rgba8();
        let mut stage_buffer = BufferRaii::new(
            &self.device.device_d,
            &self.device.allocator,
            &vk::BufferCreateInfo::default()
                .size(img_bytes.len() as _)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            MemoryLocation::CpuToGpu,
        )?;
        stage_buffer
            .mem
            .allocation
            .mapped_slice_mut()
            .with_context(|| "unable to write to stage buffer")?[..img_bytes.len()]
            .copy_from_slice(&img_bytes);
        let mut image = ImageRaii::new(
            &self.device.device_d,
            &self.device.allocator,
            &vk::ImageCreateInfo::default()
                .array_layers(1)
                .extent(vk::Extent3D {
                    width: img_obj.width(),
                    height: img_bytes.height(),
                    depth: 1,
                })
                .format(vk::Format::R8G8B8A8_UNORM)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .mip_levels(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC),
            MemoryLocation::GpuOnly,
        )?;
        let mut deferred_cb = self.get_deferred_cb()?;
        image.barrier(
            deferred_cb.command_buffer,
            ImageAccess {
                access_flags: vk::AccessFlags::TRANSFER_WRITE,
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                stage: vk::PipelineStageFlags::TRANSFER,
            },
            0..1,
            0..1,
        );
        unsafe {
            self.device.device_d.device.cmd_copy_buffer_to_image(
                deferred_cb.command_buffer,
                stage_buffer.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_extent(vk::Extent3D {
                        width: image.res.0,
                        height: image.res.1,
                        depth: image.res.2,
                    })
                    .image_subresource(image.subresource_layers(0..1, 0))],
            );
        }
        deferred_cb.preserve_buffers.push(stage_buffer);
        self.deferred_cb = Some(deferred_cb);
        self.textures.insert(path.to_string(), image);
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
        if self.per_frame_datas[self.pfd_idx].depth_image.access.layout
            != vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        {
            self.per_frame_datas[self.pfd_idx].depth_image.barrier(
                command_buffer.command_buffer,
                ImageAccess {
                    access_flags: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                },
                0..1,
                0..1,
            );
        }
        for mesh in self.meshes.values_mut() {
            mesh.update_tr_data(&self.device, command_buffer.command_buffer)
                .inspect_err(|e| eprintln!("{e}"))
                .ok();
        }
        let view = curr_frame.get_image().get_view(&ImageViewKey {
            type_: vk::ImageViewType::TYPE_2D,
            layer_range: 0..1,
            level_range: 0..1,
        })?;
        let depth_view =
            self.per_frame_datas[self.pfd_idx]
                .depth_image
                .get_view(&ImageViewKey {
                    type_: vk::ImageViewType::TYPE_2D,
                    layer_range: 0..1,
                    level_range: 0..1,
                })?;
        let cam_dset_data = self.pipeline.get_camera_uniform(
            &self.camera,
            &mut command_buffer,
            &self.device.allocator,
        )?;
        self.pipeline.begin(
            command_buffer.command_buffer,
            (curr_frame.get_image().res.0, curr_frame.get_image().res.1),
            vec![view, depth_view],
        )?;
        self.pipeline
            .bind_camera_data(&cam_dset_data, command_buffer.command_buffer);
        self.pipeline
            .draw_meshes(self.meshes.values(), command_buffer.command_buffer);
        self.pipeline.end(command_buffer.command_buffer);
        let task = self.device.run_commands(vec![command_buffer])?;
        self.device.wait_on_task(task)?;
        self.pipeline.camera_datas.push(cam_dset_data);
        self.device
            .device_d
            .instance_raii
            .window
            .pre_present_notify();
        drop(curr_frame);
        self.pfd_idx = (self.pfd_idx + 1) % self.per_frame_datas.len();
        Ok(())
    }
}
