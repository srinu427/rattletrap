use std::sync::Arc;

use ash::vk;
use include_bytes_aligned::include_bytes_aligned;
use thiserror::Error;

use crate::wrappers::{
    descriptor_set_layout::DescriptorSetLayout, logical_device::LogicalDevice, pipeline::Pipeline,
    pipeline_layout::PipelineLayout, render_pass::RenderPass, shader_module::make_shader_module,
};

static VERT_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/textured_tri_mesh.vert.spv");
static FRAG_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/textured_tri_mesh.frag.spv");

#[derive(Debug, Error)]
pub enum TexturedTriMeshInitError {
    #[error("Render pass creation error: {0}")]
    RenderPassCreateError(vk::Result),
    #[error("Descriptor set layout creation error: {0}")]
    SetLayoutCreateError(vk::Result),
    #[error("Pipeline layout creation error: {0}")]
    PipelineLayoutCreateError(vk::Result),
    #[error("Pipeline creation error: {0}")]
    PipelineCreateError(vk::Result),
}

pub struct TexturedTriMesh {
    pipeline: Arc<Pipeline>,
}

impl TexturedTriMesh {
    pub fn new(
        device: Arc<LogicalDevice>,
        max_textures: u32,
    ) -> Result<Self, TexturedTriMeshInitError> {
        let render_pass = make_render_pass(device.clone())
            .map(Arc::new)
            .map_err(TexturedTriMeshInitError::RenderPassCreateError)?;

        let set_layouts = make_set_layouts(device.clone(), max_textures)
            .map_err(TexturedTriMeshInitError::SetLayoutCreateError)?;

        let pipeline_layout = PipelineLayout::new(device.clone(), set_layouts)
            .map(Arc::new)
            .map_err(TexturedTriMeshInitError::PipelineLayoutCreateError)?;

        let pipeline = make_pipeline(pipeline_layout, render_pass)
            .map(Arc::new)
            .map_err(TexturedTriMeshInitError::PipelineCreateError)?;
        Ok(Self { pipeline })
    }
}

fn make_render_pass(device: Arc<LogicalDevice>) -> Result<RenderPass, vk::Result> {
    RenderPass::new(
        device,
        &vk::RenderPassCreateInfo2::default()
            .attachments(&[
                vk::AttachmentDescription2::default()
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                vk::AttachmentDescription2::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            ])
            .subpasses(&[vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&[vk::AttachmentReference2::default()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
                .depth_stencil_attachment(
                    &vk::AttachmentReference2::default()
                        .attachment(1)
                        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                )]),
    )
}

fn make_set_layouts(
    device: Arc<LogicalDevice>,
    max_textures: u32,
) -> Result<Vec<Arc<DescriptorSetLayout>>, vk::Result> {
    let layout0 = DescriptorSetLayout::new(
        device.clone(),
        &[
            (vk::DescriptorType::STORAGE_BUFFER, 1),
            (vk::DescriptorType::STORAGE_BUFFER, 1),
            (vk::DescriptorType::STORAGE_BUFFER, 1),
        ],
    )?;
    let layout1 = DescriptorSetLayout::new(device.clone(), &[(vk::DescriptorType::SAMPLER, 1)])?;
    let layout2 =
        DescriptorSetLayout::new(device, &[(vk::DescriptorType::SAMPLED_IMAGE, max_textures)])?;
    Ok(vec![
        Arc::new(layout0),
        Arc::new(layout1),
        Arc::new(layout2),
    ])
}

fn make_pipeline(
    layout: Arc<PipelineLayout>,
    render_pass: Arc<RenderPass>,
) -> Result<Pipeline, vk::Result> {
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend_state =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let vert_shader = make_shader_module(&render_pass.device(), VERT_SHADER_CODE)?;
    let frag_shader = make_shader_module(&render_pass.device(), FRAG_SHADER_CODE)?;

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader)
            .name(c"main"),
    ];

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .render_pass(render_pass.render_pass())
        .subpass(0)
        .layout(layout.pipeline_layout())
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .dynamic_state(&dynamic_state)
        .viewport_state(&viewport_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .rasterization_state(&rasterization_state)
        .stages(&shader_stages);

    let pipeline = unsafe {
        render_pass
            .device()
            .device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_, e)| e)?[0]
    };
    Ok(Pipeline::new(render_pass, layout, pipeline))
}
