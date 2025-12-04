use std::sync::Arc;

use ash::vk;

use crate::{
    render_objs::{Mesh, MeshTexture},
    vk_wrap::{
        device::Device,
        pipeline::{
            DSetBindingInfo, Dsl, Pipeline, PipelineError, PipelineLayout, RenderPass, ShaderModule,
        },
    },
};

#[repr(align(4))]
struct AlignedBytes<const N: usize>([u8; N]);

const VERT_SHADER_CODE: &[u8] =
    &AlignedBytes(*include_bytes!("shaders/textured_tri_mesh.vert.spv")).0;
const FRAG_SHADER_CODE: &[u8] =
    &AlignedBytes(*include_bytes!("shaders/textured_tri_mesh.frag.spv")).0;

pub struct MeshPipelineDrawable<'a> {
    mesh: &'a Mesh,
    texture: &'a MeshTexture,
}

#[derive(Debug, thiserror::Error)]
pub enum MeshPipelineError {
    #[error("Descriptor Set Layout related error: {0}")]
    PipelineError(#[from] PipelineError),
    #[error("Error creating Vulkan Shader Module: {0}")]
    ShaderModuleCreateError(vk::Result),
}

pub struct MeshPipeline {
    dsls: [Dsl; 2],
    layout: PipelineLayout,
    pipeline: Pipeline,
    render_pass: RenderPass,
}

impl MeshPipeline {
    fn make_render_pass(device: &Arc<Device>) -> Result<RenderPass, MeshPipelineError> {
        let render_pass = unsafe {
            device
                .device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[
                            vk::AttachmentDescription::default()
                                .format(vk::Format::R8G8B8A8_UNORM)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                            vk::AttachmentDescription::default()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                        ])
                        .subpasses(&[vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&[vk::AttachmentReference::default()
                                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .attachment(0)])
                            .depth_stencil_attachment(
                                &vk::AttachmentReference::default()
                                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                    .attachment(1),
                            )]),
                    None,
                )
                .map_err(PipelineError::RenderPassCreateError)?
        };
        Ok(RenderPass {
            rp: render_pass,
            device: device.clone(),
        })
    }

    fn make_pipeline(
        device: &Arc<Device>,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Pipeline, MeshPipelineError> {
        let vert = ShaderModule::new(device, &VERT_SHADER_CODE)?;
        let frag = ShaderModule::new(device, &FRAG_SHADER_CODE)?;
        let pipeline = unsafe {
            device
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .render_pass(render_pass)
                        .subpass(0)
                        .layout(layout)
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_write_enable(true)
                                .depth_test_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .sample_shading_enable(false)
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .polygon_mode(vk::PolygonMode::FILL)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .line_width(1.0),
                        )
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(c"main")
                                .module(vert.sm),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(c"main")
                                .module(frag.sm),
                        ])],
                    None,
                )
                .map_err(|(_, e)| PipelineError::PipelineCreateError(e))?[0]
        };
        drop(vert);
        drop(frag);
        Ok(pipeline)
    }

    pub fn new(device: &Arc<Device>) -> Result<Self, MeshPipelineError> {
        let render_pass = Self::make_render_pass(&device)?;
        let dsls = [
            Dsl::new(
                device,
                false,
                &[
                    DSetBindingInfo::StorageBuffer(1),
                    DSetBindingInfo::StorageBuffer(1),
                    DSetBindingInfo::StorageBuffer(1),
                ],
            )?,
            Dsl::new(device, true, &[DSetBindingInfo::Sampler2d(1000)])?,
        ];
        let layout = PipelineLayout::new(device, &dsls.iter().collect::<Vec<_>>(), 0)?;
        let pipeline = Self::make_pipeline(&device, layout.pl, render_pass.rp)?;
        let pipeline = Pipeline {
            pipeline,
            device: device.clone(),
        };
        Ok(Self {
            render_pass,
            pipeline,
            layout,
            dsls,
        })
    }

    pub fn allocate_sets(
        &self,
        device: &ash::Device,
        pool: vk::DescriptorPool,
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
        let dsls_vk: Vec<_> = self.dsls.iter().map(|d| d.dsl).collect();
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&dsls_vk),
            )?
        };
        Ok(sets)
    }
}
