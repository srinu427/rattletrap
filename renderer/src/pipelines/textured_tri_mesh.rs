use std::{
    mem,
    sync::{Arc, Mutex},
};

use ash::vk;
use bytemuck::NoUninit;
use gpu_allocator::vulkan::Allocator;
use include_bytes_aligned::include_bytes_aligned;

use anyhow::Result as AnyResult;

use crate::{
    pipelines::data_transfer::{DTPInput, DTP}, renderables::{camera::Camera, tri_mesh::TriMesh}, wrappers::{
        buffer::Buffer, descriptor_pool::DescriptorPool, descriptor_set::DescriptorSet,
        descriptor_set_layout::DescriptorSetLayout, framebuffer::Framebuffer, image::Image,
        image_view::ImageView, logical_device::LogicalDevice, pipeline::Pipeline,
        pipeline_layout::PipelineLayout, render_pass::RenderPass, sampler::Sampler,
        shader_module::make_shader_module,
    }
};

static VERT_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/textured_tri_mesh.vert.spv");
static FRAG_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/textured_tri_mesh.frag.spv");
static MAX_VERTICES: u64 = 100_000;
static MAX_MATERIALS: u64 = 10_000;

#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct MaterialInfo {
    sampler_id: u32,
    texture_id: u32,
}

pub struct TTMPSets {
    pub ssbo1: Arc<Buffer>,
    pub ssbo2: Arc<Buffer>,
    pub ssbo3: Arc<Buffer>,
    pub descriptor_sets: Vec<Arc<DescriptorSet>>,
    ttmp: Arc<TTMP>,
}

impl TTMPSets {
    pub fn new(
        ttmp: Arc<TTMP>,
        allocator: Arc<Mutex<Allocator>>,
        descriptor_pool: Arc<DescriptorPool>,
    ) -> AnyResult<Self> {
        let device = ttmp.pipeline.render_pass().device();

        // Create SSBOs and staging buffer
        let ssbo1_size = MAX_VERTICES * mem::size_of::<TriMesh>() as u64;
        let ssbo2_size = mem::size_of::<u32>() as u64;
        let ssbo3_size = MAX_MATERIALS * mem::size_of::<MaterialInfo>() as u64;

        let mut ssbo1 = Buffer::new(
            device.clone(),
            ssbo1_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        ssbo1.allocate_memory(allocator.clone(), true)?;

        let mut ssbo2 = Buffer::new(
            device.clone(),
            ssbo2_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        ssbo2.allocate_memory(allocator.clone(), true)?;

        let mut ssbo3 = Buffer::new(
            device.clone(),
            ssbo3_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        ssbo3.allocate_memory(allocator.clone(), true)?;

        // Allocate descriptor sets
        let vk_set_layouts = ttmp
            .pipeline
            .layout()
            .set_layouts()
            .iter()
            .map(|l| l.layout())
            .collect::<Vec<_>>();
        let descriptor_sets = unsafe {
            device.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool.pool())
                    .set_layouts(&vk_set_layouts),
            )?
        };
        let descriptor_sets = descriptor_sets
            .into_iter()
            .map(|ds| Arc::new(DescriptorSet::new(descriptor_pool.clone(), ds)))
            .collect::<Vec<_>>();

        unsafe {
            device.device().update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[0].set())
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(ssbo1.buffer())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[0].set())
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(ssbo2.buffer())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[0].set())
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(ssbo3.buffer())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[1].set())
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&[
                            vk::DescriptorImageInfo::default().sampler(ttmp.sampler.sampler())
                        ]),
                ],
                &[],
            );
        }

        Ok(Self {
            ssbo1: Arc::new(ssbo1),
            ssbo2: Arc::new(ssbo2),
            ssbo3: Arc::new(ssbo3),
            descriptor_sets,
            ttmp,
        })
    }

    pub fn update_textures(&self, textures: &[Arc<ImageView>]) {
        if textures.len() as u32 > self.ttmp.max_textures {
            // TODO: Handle this error properly
            panic!("Number of textures exceeds maximum supported by the pipeline");
        }

        // Update descriptor set for textures
        let device = self.ttmp.pipeline.render_pass().device();
        let image_infos: Vec<vk::DescriptorImageInfo> = textures
            .iter()
            .map(|tex| {
                vk::DescriptorImageInfo::default()
                    .image_view(tex.image_view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();

        unsafe {
            device.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[2].set())
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_infos)],
                &[],
            );
        }
    }

    pub fn update_ssbos(&self, dtp: &DTP, meshes: &[TriMesh], camera: Camera, material_infos: &[MaterialInfo]) -> AnyResult<()> {
        let vert_data: Vec<u8> = meshes
            .iter()
            .flat_map(|m| bytemuck::cast_slice(&m.vertices).to_vec())
            .collect();
        let cam_data: Vec<u8> = bytemuck::cast_slice(&[camera]).to_vec();
        let mat_data: Vec<u8> = bytemuck::cast_slice(material_infos).to_vec();

        dtp.do_transfers(vec![
            DTPInput::CopyToBuffer {
                data: &vert_data,
                buffer: &self.ssbo1,
            },
            DTPInput::CopyToBuffer {
                data: &cam_data,
                buffer: &self.ssbo2,
            },
            DTPInput::CopyToBuffer {
                data: &mat_data,
                buffer: &self.ssbo3,
            },
        ])?;
        
        Ok(())
    }
}

pub struct TTMPAttachments {
    color: Arc<ImageView>,
    depth: Arc<ImageView>,
    framebuffer: Arc<Framebuffer>,
    extent: vk::Extent2D,
    ttmp: Arc<TTMP>,
}

impl TTMPAttachments {
    pub fn new(
        ttmp: Arc<TTMP>,
        allocator: Arc<Mutex<Allocator>>,
        extent: vk::Extent2D,
    ) -> AnyResult<Self> {
        let device = ttmp.pipeline.render_pass().device();

        // Create color attachment
        let mut color_image = Image::new_2d(
            device.clone(),
            vk::Format::R8G8B8A8_SRGB,
            extent,
            1,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
        )?;
        color_image.allocate_memory(allocator.clone(), true)?;
        let color_image = Arc::new(color_image);
        let color_view = ImageView::new(
            color_image,
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .map(Arc::new)?;

        // Create depth attachment
        let mut depth_image = Image::new_2d(
            device.clone(),
            vk::Format::D32_SFLOAT,
            extent,
            1,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        depth_image.allocate_memory(allocator, true)?;
        let depth_image = Arc::new(depth_image);
        let depth_view = ImageView::new(
            depth_image,
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .map(Arc::new)?;

        // Create framebuffer
        let framebuffer = Framebuffer::new(
            ttmp.pipeline.render_pass().clone(),
            vec![color_view.clone(), depth_view.clone()],
            extent,
            1,
        )
        .map(Arc::new)?;

        Ok(Self {
            color: color_view,
            depth: depth_view,
            framebuffer,
            extent,
            ttmp,
        })
    }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct TTMP {
    #[get = "pub"]
    pipeline: Arc<Pipeline>,
    sampler: Arc<Sampler>,
    #[get_copy = "pub"]
    max_textures: u32,
}

impl TTMP {
    pub fn new(device: Arc<LogicalDevice>) -> AnyResult<Self> {
        let device_limits = unsafe {
            device
                .instance()
                .instance()
                .get_physical_device_properties(device.gpu())
        };
        let max_dset_textures = device_limits.limits.max_descriptor_set_sampled_images;
        let max_stage_textures = device_limits.limits.max_per_stage_descriptor_sampled_images;
        let max_textures = max_dset_textures.min(max_stage_textures);
        let render_pass = make_render_pass(device.clone()).map(Arc::new)?;

        let set_layouts = make_set_layouts(device.clone(), max_textures)?;

        let pipeline_layout = PipelineLayout::new(device.clone(), set_layouts).map(Arc::new)?;

        let pipeline = make_pipeline(pipeline_layout, render_pass).map(Arc::new)?;

        let sampler = Sampler::new_nearest(device.clone()).map(Arc::new)?;
        Ok(Self {
            pipeline,
            sampler,
            max_textures,
        })
    }
}

fn make_render_pass(device: Arc<LogicalDevice>) -> AnyResult<RenderPass> {
    Ok(RenderPass::new(
        device,
        &vk::RenderPassCreateInfo2::default()
            .attachments(&[
                vk::AttachmentDescription2::default()
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                vk::AttachmentDescription2::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
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
    )?)
}

fn make_set_layouts(
    device: Arc<LogicalDevice>,
    max_textures: u32,
) -> AnyResult<Vec<Arc<DescriptorSetLayout>>> {
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

fn make_pipeline(layout: Arc<PipelineLayout>, render_pass: Arc<RenderPass>) -> AnyResult<Pipeline> {
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

    // Clean up shader modules
    unsafe {
        render_pass
            .device()
            .device()
            .destroy_shader_module(vert_shader, None);
        render_pass
            .device()
            .device()
            .destroy_shader_module(frag_shader, None);
    }
    Ok(Pipeline::new(render_pass, layout, pipeline))
}
