use ash::vk;

use crate::wrappers::{
    buffer::Buffer,
    command_buffer::{CommandBuffer, CommandBufferError},
    descriptor_set::DescriptorSet,
    framebuffer::Framebuffer,
    image::{Image, ImageAccess},
    pipeline::Pipeline,
};

pub enum BarrierCommand {
    Image2d {
        image: vk::Image,
        format: vk::Format,
        subresource_range: vk::ImageSubresourceRange,
        old_access: ImageAccess,
        new_access: ImageAccess,
    },
    Buffer {
        buffer: vk::Buffer,
        old_access: vk::AccessFlags2,
        new_access: vk::AccessFlags2,
        old_stage: vk::PipelineStageFlags2,
        new_stage: vk::PipelineStageFlags2,
    }
}

impl BarrierCommand {
    pub fn new_image_2d_barrier(
        image: &Image,
        old_access: ImageAccess,
        new_access: ImageAccess,
    ) -> Self {
        Self::Image2d {
            image: image.image(),
            format: image.format(),
            subresource_range: image.full_subresource_range(),
            old_access,
            new_access,
        }
    }

    pub fn apply_command(&self, cmd_buffer: &CommandBuffer) {
        let device = cmd_buffer.command_pool().device().sync2_device();
        match self {
            BarrierCommand::Image2d {
                image,
                format,
                subresource_range,
                old_access,
                new_access,
            } => {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(old_access.to_stage_flags(*format))
                    .src_access_mask(old_access.to_access_flags(*format))
                    .dst_stage_mask(new_access.to_stage_flags(*format))
                    .dst_access_mask(new_access.to_access_flags(*format))
                    .old_layout(old_access.to_layout(*format))
                    .new_layout(new_access.to_layout(*format))
                    .image(*image)
                    .subresource_range(*subresource_range);

                unsafe {
                    device.cmd_pipeline_barrier2(
                        cmd_buffer.command_buffer(),
                        &vk::DependencyInfo::default()
                            .dependency_flags(vk::DependencyFlags::BY_REGION)
                            .image_memory_barriers(std::slice::from_ref(&barrier)),
                    );
                }
            },
            BarrierCommand::Buffer {
                buffer,
                old_access,
                new_access,
                old_stage,
                new_stage,
            } => {
                let barrier = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(*old_stage)
                    .src_access_mask(*old_access)
                    .dst_stage_mask(*new_stage)
                    .dst_access_mask(*new_access)
                    .buffer(*buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);

                unsafe {
                    device.cmd_pipeline_barrier2(
                        cmd_buffer.command_buffer(),
                        &vk::DependencyInfo::default()
                            .dependency_flags(vk::DependencyFlags::BY_REGION)
                            .buffer_memory_barriers(std::slice::from_ref(&barrier)),
                    );
                }
            }
        }
    }
}

pub enum RenderCommand {
    BindPipeline(usize),
    BindDescriptorSets {
        pipeline_id: usize,
        sets: Vec<usize>,
    },
    Draw(u32),
}

impl RenderCommand {
    pub fn record(
        &self,
        cmd_buffer: &CommandBuffer,
        pipelines: &[vk::Pipeline],
        pipeline_layouts: &[vk::PipelineLayout],
        dsets: &[vk::DescriptorSet],
    ) {
        let device = cmd_buffer.command_pool().device().device();
        match self {
            RenderCommand::BindPipeline(index) => {
                let pipeline = pipelines[*index as usize];
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd_buffer.command_buffer(),
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                }
            }
            RenderCommand::BindDescriptorSets { pipeline_id, sets } => {
                let vk_sets: Vec<vk::DescriptorSet> =
                    sets.iter().map(|&i| dsets[i as usize]).collect();
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd_buffer.command_buffer(),
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layouts[*pipeline_id],
                        0,
                        &vk_sets,
                        &[],
                    );
                }
            }
            RenderCommand::Draw(vertex_count) => unsafe {
                device.cmd_draw(cmd_buffer.command_buffer(), *vertex_count, 1, 0, 0);
            },
        }
    }
}

pub enum Command {
    CopyBufferToBuffer {
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyBufferToImage {
        src: vk::Buffer,
        dst: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
    },
    BlitImage {
        src: vk::Image,
        dst: vk::Image,
        filter: vk::Filter,
        regions: Vec<vk::ImageBlit>,
    },
    RunRenderPass {
        render_pass: vk::RenderPass,
        pipelines: Vec<vk::Pipeline>,
        pipeline_layouts: Vec<vk::PipelineLayout>,
        dsets: Vec<vk::DescriptorSet>,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
        clear_values: Vec<vk::ClearValue>,
        commands: Vec<RenderCommand>,
    },
    Barrier(BarrierCommand),
}

impl Command {
    pub fn copy_buffer_to_buffer(src: &Buffer, dst: &Buffer, regions: Vec<vk::BufferCopy>) -> Self {
        Self::CopyBufferToBuffer {
            src: src.buffer(),
            dst: dst.buffer(),
            regions,
        }
    }

    pub fn copy_buffer_to_image(
        src: &Buffer,
        dst: &Image,
        regions: Vec<vk::BufferImageCopy>,
    ) -> Self {
        Self::CopyBufferToImage {
            src: src.buffer(),
            dst: dst.image(),
            regions,
        }
    }

    pub fn blit_full_image(src: &Image, dst: &Image, filter: vk::Filter) -> Self {
        let blit_region = vk::ImageBlit::default()
            .src_subresource(src.all_subresource_layers(0))
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src.extent().width as i32,
                    y: src.extent().height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst.all_subresource_layers(0))
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst.extent().width as i32,
                    y: dst.extent().height as i32,
                    z: 1,
                },
            ]);
        Self::BlitImage {
            src: src.image(),
            dst: dst.image(),
            filter,
            regions: vec![blit_region],
        }
    }

    pub fn run_render_pass(
        pipelines: Vec<&Pipeline>,
        dsets: Vec<&DescriptorSet>,
        framebuffer: &Framebuffer,
        clear_values: Vec<vk::ClearValue>,
        commands: Vec<RenderCommand>,
    ) -> Self {
        Self::RunRenderPass {
            render_pass: pipelines[0].render_pass().render_pass(),
            pipelines: pipelines.iter().map(|p| p.pipeline()).collect(),
            pipeline_layouts: pipelines
                .iter()
                .map(|pl| pl.layout().pipeline_layout())
                .collect(),
            dsets: dsets.iter().map(|ds| ds.set()).collect(),
            framebuffer: framebuffer.framebuffer(),
            extent: framebuffer.extent(),
            clear_values,
            commands,
        }
    }

    pub fn record(&self, cmd_buffer: &CommandBuffer) {
        let device = cmd_buffer.command_pool().device();
        match self {
            Self::CopyBufferToBuffer { src, dst, regions } => unsafe {
                device
                    .device()
                    .cmd_copy_buffer(cmd_buffer.command_buffer(), *src, *dst, regions);
            },
            Self::CopyBufferToImage { src, dst, regions } => unsafe {
                device.device().cmd_copy_buffer_to_image(
                    cmd_buffer.command_buffer(),
                    *src,
                    *dst,
                    ImageAccess::TransferDst.to_layout(vk::Format::UNDEFINED),
                    regions,
                );
            },
            Self::BlitImage {
                src,
                dst,
                filter,
                regions,
            } => unsafe {
                device.device().cmd_blit_image(
                    cmd_buffer.command_buffer(),
                    *src,
                    ImageAccess::TransferSrc.to_layout(vk::Format::UNDEFINED),
                    *dst,
                    ImageAccess::TransferDst.to_layout(vk::Format::UNDEFINED),
                    &regions,
                    *filter,
                );
            },
            Self::RunRenderPass {
                pipelines,
                pipeline_layouts,
                dsets,
                framebuffer,
                commands,
                clear_values,
                render_pass,
                extent,
            } => {
                let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                    .render_pass(*render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: *extent,
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
                            width: extent.width as f32,
                            height: extent.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );

                    device.device().cmd_set_scissor(
                        cmd_buffer.command_buffer(),
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: *extent,
                        }],
                    );

                    for command in commands {
                        command.record(&cmd_buffer, &pipelines, &pipeline_layouts, &dsets);
                    }
                    device
                        .device()
                        .cmd_end_render_pass(cmd_buffer.command_buffer());
                }
            }
            Self::Barrier(barrier_command) => {
                barrier_command.apply_command(cmd_buffer);
            }
        }
    }
}

impl CommandBuffer {
    pub fn record_commands(&self, commands: &[Command], one_time: bool) -> Result<(), CommandBufferError> {
        self.begin(one_time)?;
        for command in commands {
            command.record(self);
        }
        self.end()?;
        Ok(())
    }
}
