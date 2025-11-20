use ash::vk;

use crate::{make_init_struct_copy, vk12::image_vk::image_subresource_layers_2d};

make_init_struct_copy!(
    InitCommandPool,
    vk::CommandPool,
    self,
    self.device.destroy_command_pool(self.inner, None)
);

pub fn create_command_pool(
    device: &'_ ash::Device,
    queue_family: u32,
) -> Result<InitCommandPool<'_>, vk::Result> {
    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family),
            None,
        )?
    };
    Ok(InitCommandPool {
        drop: true,
        inner: command_pool,
        device,
    })
}

pub fn allocate_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(count),
        )
    }
}

pub fn begin_cmd_buffer(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    one_time: bool,
) -> Result<(), vk::Result> {
    let flags = if one_time {
        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
    } else {
        vk::CommandBufferUsageFlags::empty()
    };
    unsafe {
        device.begin_command_buffer(
            cmd_buffer,
            &vk::CommandBufferBeginInfo::default().flags(flags),
        )?;
    }

    Ok(())
}

pub fn end_cmd_buffer(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
) -> Result<(), vk::Result> {
    unsafe {
        device.end_command_buffer(cmd_buffer)?;
    }
    Ok(())
}

pub struct CompositeInput {
    pub image: vk::Image,
    pub image_res: vk::Extent2D,
    pub in_range: [(f32, f32); 2],
    pub out_range: [(f32, f32); 2],
}

pub fn composite_images(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    dst: vk::Image,
    dst_res: vk::Extent2D,
    inputs: Vec<CompositeInput>,
) {
    unsafe {
        for inp in inputs {
            let src_offsets = [
                vk::Offset3D::default()
                    .x((inp.in_range[0].0 * inp.image_res.width as f32) as _)
                    .y((inp.in_range[0].1 * inp.image_res.height as f32) as _),
                vk::Offset3D::default()
                    .x((inp.in_range[1].0 * inp.image_res.width as f32) as _)
                    .y((inp.in_range[1].1 * inp.image_res.height as f32) as _)
                    .z(1),
            ];
            let dst_offsets = [
                vk::Offset3D::default()
                    .x((inp.out_range[0].0 * dst_res.width as f32) as _)
                    .y((inp.out_range[0].1 * dst_res.height as f32) as _),
                vk::Offset3D::default()
                    .x((inp.out_range[1].0 * dst_res.width as f32) as _)
                    .y((inp.out_range[1].1 * dst_res.height as f32) as _)
                    .z(1),
            ];
            device.cmd_blit_image(
                cmd_buffer,
                inp.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(image_subresource_layers_2d(false, false))
                    .src_offsets(src_offsets)
                    .dst_subresource(image_subresource_layers_2d(false, false))
                    .dst_offsets(dst_offsets)],
                vk::Filter::NEAREST,
            );
        }
    }
}
