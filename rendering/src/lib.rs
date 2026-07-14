use std::sync::Arc;

use anyhow::Context;
use ash::vk;
use gpu_allocator::MemoryLocation;
use indexmap::IndexMap;
use winit::window::Window;

use crate::vkraii::{
    command::CommandBufferRaii,
    device::DeviceRaii,
    resource::{BufferRaii, ImageAccess, ImageRaii},
    swapchain::SwapchainRaii,
};

pub mod tex_mesh;
mod vkraii;

pub struct RenderingManager {
    textures: IndexMap<String, ImageRaii>,
    deferred_cb: Option<CommandBufferRaii>,
    swapchain: SwapchainRaii,
    device: DeviceRaii,
    window: Arc<Window>,
}

impl RenderingManager {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let device = DeviceRaii::new(window)?;
        let swapchain = SwapchainRaii::new(&device.device_d)?;
        Ok(Self {
            textures: Default::default(),
            deferred_cb: None,
            swapchain,
            device,
            window: window.clone(),
        })
    }

    pub fn refresh_size(&mut self) -> anyhow::Result<()> {
        self.swapchain.refresh()
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
        let mut semaphore = self.device.sync_pool.get_sem()?;
        if let Some(deferred_cb) = self.deferred_cb.take() {
            self.device
                .run_commands(vec![deferred_cb], &mut semaphore)?
                .wait()?;
        }
        let mut command_buffer = self.device.command_pool.get_cb()?;
        command_buffer.begin()?;
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
        self.device
            .run_commands(vec![command_buffer], &mut semaphore)?
            .wait()?;
        drop(curr_frame);
        Ok(())
    }
}
