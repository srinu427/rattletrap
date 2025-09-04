use ash::vk;

use crate::wrappers::{buffer::Buffer, command, command_buffer::CommandBuffer, descriptor_set::DescriptorSet, framebuffer::Framebuffer, image::Image, pipeline::Pipeline, render_pass::RenderPass};

pub enum BarrierCommand<'a> {
    Image2d {
        image: &'a Image,
        old_layout: vk::ImageLayout,
        old_stage: vk::PipelineStageFlags2,
        old_access: vk::AccessFlags2,
        new_layout: vk::ImageLayout,
        new_stage: vk::PipelineStageFlags2,
        new_access: vk::AccessFlags2,
        aspect_mask: vk::ImageAspectFlags,
    }
}

impl<'a> BarrierCommand<'a> {
    pub fn apply_command(&self, cmd_buffer: &CommandBuffer) {
        let device = cmd_buffer.command_pool().device();
        match self {
            BarrierCommand::Image2d {
                image,
                old_layout,
                old_stage,
                old_access,
                new_layout,
                new_stage,
                new_access,
                aspect_mask,
            } => {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(*old_stage)
                    .src_access_mask(*old_access)
                    .dst_stage_mask(*new_stage)
                    .dst_access_mask(*new_access)
                    .old_layout(*old_layout)
                    .new_layout(*new_layout)
                    .image(image.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(*aspect_mask)
                            .base_mip_level(0)
                            .level_count(image.mip_levels())
                            .base_array_layer(0)
                            .layer_count(image.array_layers()),
                    );
                
                unsafe {
                    device.sync2_device().cmd_pipeline_barrier2(
                        cmd_buffer.command_buffer(),
                        &vk::DependencyInfo::default()
                            .image_memory_barriers(std::slice::from_ref(&barrier)),
                    );
                }
            }
        }
    }
}

pub enum RenderCommand {
    BindPipeline(u32),
    BindDescriptorSets(Vec<u32>),
    Draw(u32),
}

impl RenderCommand {
    pub fn record(&self, cmd_buffer: &CommandBuffer, pipelines: &[&Pipeline], dsets: &[&DescriptorSet]) {
        let device = cmd_buffer.command_pool().device();
        match self {
            RenderCommand::BindPipeline(index) => {
                let pipeline = pipelines[*index as usize];
                unsafe {
                    device.device().cmd_bind_pipeline(
                        cmd_buffer.command_buffer(),
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline(),
                    );
                }
            }
            RenderCommand::BindDescriptorSets(indices) => {
                let sets: Vec<vk::DescriptorSet> = indices.iter().map(|&i| dsets[i as usize].set()).collect();
                unsafe {
                    device.device().cmd_bind_descriptor_sets(
                        cmd_buffer.command_buffer(),
                        vk::PipelineBindPoint::GRAPHICS,
                        pipelines[0].layout().pipeline_layout(),
                        0,
                        &sets,
                        &[],
                    );
                }
            }
            RenderCommand::Draw(vertex_count) => {
                unsafe {
                    device.device().cmd_draw(
                        cmd_buffer.command_buffer(),
                        *vertex_count,
                        1,
                        0,
                        0,
                    );
                }
            }
        }
    }
}

pub enum Command<'a> {
    CopyBufferToBuffer {
        src: &'a Buffer,
        dst: &'a Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyBufferToImage {
        src: &'a Buffer,
        dst: &'a Image,
        regions: Vec<vk::BufferImageCopy>,
    },
    RunRenderPass {
        pipelines: Vec<&'a Pipeline>,
        dsets: Vec<&'a DescriptorSet>,
        framebuffer: &'a Framebuffer,
        clear_values: Vec<vk::ClearValue>,
        commands: Vec<RenderCommand>,
    },
    Barrier(BarrierCommand<'a>),
}

impl<'a> Command<'a> {
    pub fn record(&self, cmd_buffer: &CommandBuffer) {
        let device = cmd_buffer.command_pool().device();
        match self {
            Self::CopyBufferToBuffer { src, dst, regions } => {
                unsafe {
                    device.device().cmd_copy_buffer(
                        cmd_buffer.command_buffer(),
                        src.buffer(),
                        dst.buffer(),
                        regions,
                    );
                }
            }
            Self::CopyBufferToImage { src, dst, regions } => {
                unsafe {
                    device.device().cmd_copy_buffer_to_image(
                        cmd_buffer.command_buffer(),
                        src.buffer(),
                        dst.image(),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        regions,
                    );
                }
            }
            Self::RunRenderPass { pipelines, dsets, framebuffer, commands, clear_values } => {
                let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                    .render_pass(framebuffer.render_pass().render_pass())
                    .framebuffer(framebuffer.framebuffer())
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: framebuffer.extent(),
                    })
                    .clear_values(&clear_values);
                unsafe {
                    device.device().cmd_begin_render_pass(
                        cmd_buffer.command_buffer(),
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );

                    device.device().cmd_set_viewport(
                        cmd_buffer.command_buffer(),
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: framebuffer.extent().width as f32,
                            height: framebuffer.extent().height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );

                    device.device().cmd_set_scissor(
                        cmd_buffer.command_buffer(),
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: framebuffer.extent(),
                        }],
                    );

                    for command in commands {
                        command.record(cmd_buffer, &pipelines, &dsets);
                    }
                    device.device().cmd_end_render_pass(cmd_buffer.command_buffer());
                }
            }
            Self::Barrier(barrier_command) => {
                barrier_command.apply_command(cmd_buffer);
            },
        }
    }
}