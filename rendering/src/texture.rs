use anyhow::Context;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{
    GpuClient,
    utils::{ImageAccess, StagingBuffer, create_image},
};

pub struct Texture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    mem: Allocation,
}

impl Texture {
    pub fn load_image(
        client: &mut GpuClient,
        path: &str,
        usage: vk::ImageUsageFlags,
        dst_image_access: ImageAccess,
    ) -> anyhow::Result<Self> {
        let image_data = image::open(path)?;
        let image_bytes = image_data.to_rgba8();
        let image_bytes_len = image_bytes.len();
        let (image, allocation) = create_image(
            &client.device,
            &mut client.allocator,
            &vk::ImageCreateInfo::default()
                .array_layers(1)
                .extent(vk::Extent3D {
                    width: image_data.width(),
                    height: image_data.height(),
                    depth: 1,
                })
                .format(vk::Format::R8G8B8A8_UNORM)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .mip_levels(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(usage),
        )?;
        let mut staging_buffer =
            StagingBuffer::new(&client.device, &mut client.allocator, image_bytes_len as _)?;
        staging_buffer
            .mem
            .mapped_slice_mut()
            .context("cant get mapped memory of staged buffer")?[..image_bytes_len]
            .copy_from_slice(&image_bytes);
        let command_buffer = client.get_deferred_cb()?;
        unsafe {
            client.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_queue_family_index(client.graphics_qf)
                    .image(image)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .src_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(client.graphics_qf)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )],
            );
            client.device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_extent(vk::Extent3D {
                        width: image_data.width(),
                        height: image_data.height(),
                        depth: 1,
                    })
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )],
            );
            client.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                dst_image_access.access_stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(dst_image_access.access_flags)
                    .dst_queue_family_index(client.graphics_qf)
                    .image(image)
                    .new_layout(dst_image_access.layout)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(client.graphics_qf)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )],
            );
        }
        let view = unsafe {
            client.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .components(vk::ComponentMapping::default())
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )
                    .view_type(vk::ImageViewType::TYPE_2D),
                None,
            )?
        };
        client.deferred_preserve_buffers.push(staging_buffer);
        Ok(Self {
            image,
            view,
            mem: allocation,
        })
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            allocator
                .free(self.mem)
                .inspect_err(|e| log::warn!("freeing memory of image {:?} failed: {e}", self.image))
                .ok();
        }
    }
}
