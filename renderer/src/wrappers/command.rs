use std::sync::Arc;

use ash::vk;

use crate::wrappers::{buffer::Buffer, command_buffer::CommandBuffer, framebuffer::Framebuffer, image::Image, image_access::ImageAccess};

pub enum RenderPassCommand {
    BindPipeline {
        pipeline: usize,
    },
    BindShaderInput {
        pipeline_layout: usize,
        descriptor_sets: Vec<vk::DescriptorSet>,
    },
    BindVertexBuffers {
        buffers: Vec<Arc<Buffer>>,
    },
    BindIndexBuffer {
        buffer: Arc<Buffer>,
    },
    SetPushConstant {
        pipeline_layout: usize,
        data: Vec<u8>,
    },
    Draw {
        count: u32,
        vertex_offset: i32,
        index_offset: u32,
    },
}

impl RenderPassCommand {
    pub fn apply_command(
        &self,
        command_buffer: &CommandBuffer,
        pipelines: &[vk::Pipeline],
        pipeline_layouts: &[vk::PipelineLayout],
    ) {
        let device = command_buffer.command_pool().device().device();
        let vk_command_buffer = command_buffer.command_buffer();
        unsafe {
            match self {
                RenderPassCommand::BindPipeline { pipeline } => {
                    device.cmd_bind_pipeline(
                        vk_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipelines[*pipeline],
                    );
                }
                RenderPassCommand::BindShaderInput {
                    pipeline_layout,
                    descriptor_sets,
                } => {
                    device.cmd_bind_descriptor_sets(
                        vk_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layouts[*pipeline_layout],
                        0,
                        descriptor_sets,
                        &[],
                    );
                }
                RenderPassCommand::BindVertexBuffers { buffers } => {
                    let buffers = buffers
                        .iter()
                        .map(|b| b.buffer())
                        .collect::<Vec<_>>();
                    let offsets = vec![0; buffers.len()];
                    device.cmd_bind_vertex_buffers(
                        vk_command_buffer,
                        0,
                        &buffers,
                        &offsets,
                    );
                }
                RenderPassCommand::BindIndexBuffer { buffer } => {
                    device.cmd_bind_index_buffer(
                        vk_command_buffer,
                        buffer.buffer(),
                        0,
                        vk::IndexType::UINT32,
                    );
                }
                RenderPassCommand::SetPushConstant {
                    pipeline_layout,
                    data,
                } => {
                    device.cmd_push_constants(
                        vk_command_buffer,
                        pipeline_layouts[*pipeline_layout],
                        vk::ShaderStageFlags::ALL,
                        0,
                        data,
                    );
                }
                RenderPassCommand::Draw {
                    count,
                    vertex_offset,
                    index_offset,
                } => {
                    device.cmd_draw_indexed(
                        vk_command_buffer,
                        *count,
                        1,
                        *index_offset,
                        *vertex_offset,
                        0,
                    );
                }
            }
        }
    }
}

pub enum Command {
    ImageAccessInit {
        image: Arc<Image>,
        access: ImageAccess,
    },
    ImageAccessHint {
        image: Arc<Image>,
        access: ImageAccess,
    },
    BlitFullImage {
        src: Arc<Image>,
        dst: Arc<Image>,
    },
    RunRenderPass {
        render_pass: vk::RenderPass,
        framebuffer: Arc<Framebuffer>,
        clear_values: Vec<vk::ClearValue>,
        pipelines: Vec<vk::Pipeline>,
        pipeline_layouts: Vec<vk::PipelineLayout>,
        commands: Vec<RenderPassCommand>,
    },
    CopyBufferToImageComplete {
        buffer: Arc<Buffer>,
        image: Arc<Image>,
    },
}
