use std::{fs, sync::Arc};

use ash::vk;
use hashbrown::HashMap;
use rhi2::image::ImageView as _;

use crate::{
    buffer::Buffer,
    device::DeviceDropper,
    image::{Image, ImageView, rhi2_fmt_to_vk_fmt},
    init_helpers,
    shader::ShaderSet,
};

const MAX_CACHED_FBS: usize = 128;

pub struct DPool {
    pub handle: vk::DescriptorPool,
    pub device_dropper: Arc<DeviceDropper>,
}

impl Drop for DPool {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .device
                .destroy_descriptor_pool(self.handle, None);
        }
    }
}

fn vk_desc_type_from_dp_size(ssi: &rhi2::shader::ShaderSetInfo) -> vk::DescriptorPoolSize {
    match ssi {
        rhi2::shader::ShaderSetInfo::UniformBuffer(count) => vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(*count as _),
        rhi2::shader::ShaderSetInfo::StorageBuffer(count) => vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(*count as _),
        rhi2::shader::ShaderSetInfo::Sampler2D(count) => vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(*count as _),
    }
}

fn vk_ssi_to_dtype(ssi: &rhi2::shader::ShaderSetInfo) -> vk::DescriptorType {
    match ssi {
        rhi2::shader::ShaderSetInfo::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
        rhi2::shader::ShaderSetInfo::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
        rhi2::shader::ShaderSetInfo::Sampler2D(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    }
}

fn vk_ssi_to_dcount(ssi: &rhi2::shader::ShaderSetInfo) -> u32 {
    (match ssi {
        rhi2::shader::ShaderSetInfo::UniformBuffer(c) => *c,
        rhi2::shader::ShaderSetInfo::StorageBuffer(c) => *c,
        rhi2::shader::ShaderSetInfo::Sampler2D(c) => *c,
    }) as u32
}

pub struct DescriptorGen {
    pub bindings: Vec<rhi2::shader::ShaderSetInfo>,
    pub layout: vk::DescriptorSetLayout,
    pub pool: Arc<DPool>,
}

impl DescriptorGen {
    pub fn new(
        device_dropper: &Arc<DeviceDropper>,
        bindings: Vec<rhi2::shader::ShaderSetInfo>,
    ) -> Result<Self, String> {
        let dsl_bindings: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .descriptor_type(vk_ssi_to_dtype(b))
                    .descriptor_count(vk_ssi_to_dcount(b))
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings);
        let layout = unsafe {
            device_dropper
                .device
                .create_descriptor_set_layout(&dsl_info, None)
                .map_err(|e| format!("error creating dsl: {e}"))?
        };
        let pool_sizes: Vec<_> = bindings
            .iter()
            .map(|b| vk_desc_type_from_dp_size(b))
            .collect();
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&pool_sizes)
            .max_sets(128);
        let pool = unsafe {
            device_dropper
                .device
                .create_descriptor_pool(&pool_create_info, None)
                .map_err(|e| format!("create Desc Pool failed: {e}"))?
        };
        let pool = DPool {
            handle: pool,
            device_dropper: device_dropper.clone(),
        };
        Ok(DescriptorGen {
            bindings,
            layout,
            pool: Arc::new(pool),
        })
    }

    pub fn new_shader_set(&mut self) -> Result<ShaderSet, String> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool.handle)
            .set_layouts(core::slice::from_ref(&self.layout));
        let dset = unsafe {
            match self
                .pool
                .device_dropper
                .device
                .allocate_descriptor_sets(&alloc_info)
            {
                Ok(mut dset) => dset.remove(0),
                Err(e) => match e {
                    vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                        let pool_sizes: Vec<_> = self
                            .bindings
                            .iter()
                            .map(|b| vk_desc_type_from_dp_size(b))
                            .collect();
                        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                            .pool_sizes(&pool_sizes)
                            .max_sets(128);
                        let pool = self
                            .pool
                            .device_dropper
                            .device
                            .create_descriptor_pool(&pool_create_info, None)
                            .map_err(|e| format!("create Desc Pool failed: {e}"))?;
                        let pool = DPool {
                            handle: pool,
                            device_dropper: self.pool.device_dropper.clone(),
                        };
                        self.pool = Arc::new(pool);
                        let dset = self
                            .pool
                            .device_dropper
                            .device
                            .allocate_descriptor_sets(&alloc_info)
                            .map_err(|e| format!("create Desc Set failed: {e}"))?
                            .remove(0);
                        dset
                    }
                    _ => return Err(format!("create Desc Set failed: {e}")),
                },
            }
        };
        let dset = ShaderSet {
            handle: dset,
            pool: self.pool.clone(),
        };
        Ok(dset)
    }
}

fn store_op_vk(store: bool) -> vk::AttachmentStoreOp {
    if store {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    }
}

fn clear_op_vk(clear: bool) -> vk::AttachmentLoadOp {
    if clear {
        vk::AttachmentLoadOp::CLEAR
    } else {
        vk::AttachmentLoadOp::DONT_CARE
    }
}

fn rhi2_att_info_to_layout(desc: &rhi2::graphics_pipeline::AttachInfo) -> vk::ImageLayout {
    if desc.format.is_depth() {
        if desc.format.has_stencil() {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        }
    } else {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    }
}

fn rhi2_att_info_to_vk(desc: &rhi2::graphics_pipeline::AttachInfo) -> vk::AttachmentDescription {
    let layout = rhi2_att_info_to_layout(desc);
    vk::AttachmentDescription::default()
        .format(rhi2_fmt_to_vk_fmt(desc.format))
        .initial_layout(layout)
        .final_layout(layout)
        .load_op(clear_op_vk(desc.clear))
        .store_op(store_op_vk(desc.store))
        .samples(vk::SampleCountFlags::TYPE_1)
}

fn rhi2_va_to_vk_fmt(att: &rhi2::graphics_pipeline::VertexAttribute) -> vk::Format {
    match att {
        rhi2::graphics_pipeline::VertexAttribute::Vec3 => vk::Format::R32G32B32_SFLOAT,
        rhi2::graphics_pipeline::VertexAttribute::Vec4 => vk::Format::R32G32B32A32_SFLOAT,
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ImagesHashable {
    color: Vec<vk::ImageView>,
    depth: Option<vk::ImageView>,
}

pub struct GraphicsPipeline {
    pub handle: vk::Pipeline,
    pub layout_handle: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub descriptor_gens: Vec<DescriptorGen>,
    pub pc_size: usize,
    pub frag_output_infos: Vec<rhi2::graphics_pipeline::AttachInfo>,
    pub frag_depth_infos: Option<rhi2::graphics_pipeline::AttachInfo>,
    pub framebuffer_cache: HashMap<ImagesHashable, vk::Framebuffer>,
    pub device_dropper: Arc<DeviceDropper>,
}

impl GraphicsPipeline {
    fn create_render_pass(
        device_dropper: &Arc<DeviceDropper>,
        frag_stage_info: &rhi2::graphics_pipeline::FragmentStageInfo,
    ) -> Result<vk::RenderPass, String> {
        let mut attach_descs: Vec<_> = frag_stage_info
            .outputs
            .iter()
            .map(|a| rhi2_att_info_to_vk(a))
            .collect();
        if let Some(depth_desc) = frag_stage_info.depth {
            attach_descs.push(rhi2_att_info_to_vk(&depth_desc));
        }
        let color_refs: Vec<_> = frag_stage_info
            .outputs
            .iter()
            .enumerate()
            .map(|(i, a)| {
                vk::AttachmentReference::default()
                    .attachment(i as _)
                    .layout(rhi2_att_info_to_layout(a))
            })
            .collect();
        let depth_ref = frag_stage_info.depth.as_ref().map(|a| {
            vk::AttachmentReference::default()
                .attachment(frag_stage_info.outputs.len() as _)
                .layout(rhi2_att_info_to_layout(a))
        });
        let mut sp_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        if let Some(dref) = depth_ref.as_ref() {
            sp_desc = sp_desc.depth_stencil_attachment(dref);
        }
        let rp_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attach_descs)
            .subpasses(core::slice::from_ref(&sp_desc));
        let render_pass = unsafe {
            device_dropper
                .device
                .create_render_pass(&rp_create_info, None)
                .map_err(|e| format!("render pass creation failed: {e}"))?
        };
        Ok(render_pass)
    }

    fn load_wgsl(
        device_dropper: &Arc<DeviceDropper>,
        path: &str,
        vs_main: &str,
        fs_main: &str,
    ) -> Result<(vk::ShaderModule, vk::ShaderModule), String> {
        let wgsl_str =
            fs::read_to_string(path).map_err(|e| format!("error reading {path:?}: {e}"))?;
        let naga_mod = naga::front::wgsl::parse_str(&wgsl_str)
            .map_err(|e| format!("invalid wgsl code in {path:?}: {e}"))?;
        let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&naga_mod)
        .map_err(|e| format!("error validating shader module code: {e}"))?;
        let vert_spirv_code = naga::back::spv::write_vec(
            &naga_mod,
            &module_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Vertex,
                entry_point: vs_main.to_string(),
            }),
        )
        .map_err(|e| format!("shader conversion to spv failed: {e}"))?;
        let frag_spirv_code = naga::back::spv::write_vec(
            &naga_mod,
            &module_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Fragment,
                entry_point: fs_main.to_string(),
            }),
        )
        .map_err(|e| format!("shader conversion to spv failed: {e}"))?;
        let vert_mod = unsafe {
            device_dropper
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&vert_spirv_code),
                    None,
                )
                .map_err(|e| format!("vk shader mod creation failed: {e}"))?
        };
        let frag_mod = unsafe {
            device_dropper
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&frag_spirv_code),
                    None,
                )
                .map_err(|e| format!("vk shader mod creation failed: {e}"))?
        };
        Ok((vert_mod, frag_mod))
    }

    pub fn new(
        device_dropper: &Arc<DeviceDropper>,
        shader: &str,
        sets: Vec<Vec<rhi2::shader::ShaderSetInfo>>,
        pc_size: usize,
        vert_stage_info: rhi2::graphics_pipeline::VertexStageInfo,
        frag_stage_info: rhi2::graphics_pipeline::FragmentStageInfo,
    ) -> Result<Self, String> {
        let render_pass = Self::create_render_pass(device_dropper, &frag_stage_info)?;
        let descriptor_gens: Vec<_> = sets
            .into_iter()
            .map(|s| DescriptorGen::new(device_dropper, s))
            .collect::<Result<_, _>>()?;
        let dsls: Vec<_> = descriptor_gens.iter().map(|d| d.layout).collect();
        let pc_info = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(pc_size as _);
        let mut pl_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls);
        if pc_size > 0 {
            pl_create_info = pl_create_info.push_constant_ranges(core::slice::from_ref(&pc_info));
        }
        let pipeline_layout = unsafe {
            device_dropper
                .device
                .create_pipeline_layout(&pl_create_info, None)
                .map_err(|e| format!("pipeline layout creation failed: {e}"))?
        };
        let dyn_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
        let vp_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let vert_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(vert_stage_info.stride as _);
        let mut vert_attribs = vec![];
        let mut offset = 0;
        for (i, vsa) in vert_stage_info.attribs.iter().enumerate() {
            let va = vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(i as _)
                .offset(offset)
                .format(rhi2_va_to_vk_fmt(vsa));
            vert_attribs.push(va);
            offset += vsa.size();
        }
        let vert_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vert_attribs)
            .vertex_binding_descriptions(core::slice::from_ref(&vert_binding));
        let inp_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vs_main_name =
            init_helpers::safe_str_to_cstring(vert_stage_info.entrypoint.to_string());
        let fs_main_name =
            init_helpers::safe_str_to_cstring(frag_stage_info.entrypoint.to_string());
        let (vs_mod, fs_mod) = Self::load_wgsl(
            device_dropper,
            shader,
            vert_stage_info.entrypoint,
            frag_stage_info.entrypoint,
        )?;
        let pipeline_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_mod)
                .name(&vs_main_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_mod)
                .name(&fs_main_name),
        ];
        let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .depth_bias_enable(false);
        let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attach_blend_state: Vec<_> = (0..frag_stage_info.outputs.len())
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&attach_blend_state);
        let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut p_create_info = vk::GraphicsPipelineCreateInfo::default()
            .render_pass(render_pass)
            .dynamic_state(&dyn_info)
            .viewport_state(&vp_state)
            .layout(pipeline_layout)
            .stages(&pipeline_stages)
            .vertex_input_state(&vert_state)
            .input_assembly_state(&inp_assembly)
            .rasterization_state(&raster_state)
            .multisample_state(&msaa_state)
            .color_blend_state(&color_blending);
        if let Some(_) = &frag_stage_info.depth {
            depth_state = depth_state
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            p_create_info = p_create_info.depth_stencil_state(&depth_state);
        }
        let pipeline = unsafe {
            device_dropper
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[p_create_info], None)
                .map_err(|(_, e)| format!("vk pipeline creation failed: {e}"))?
                .remove(0)
        };
        Ok(Self {
            handle: pipeline,
            layout_handle: pipeline_layout,
            render_pass,
            descriptor_gens,
            pc_size,
            frag_output_infos: frag_stage_info.outputs,
            frag_depth_infos: frag_stage_info.depth,
            framebuffer_cache: HashMap::new(),
            device_dropper: device_dropper.clone(),
        })
    }

    pub fn create_framebuffer(
        &mut self,
        color: &[ImageView],
        depth: Option<&ImageView>,
    ) -> Result<vk::Framebuffer, String> {
        let hash_obj = ImagesHashable {
            color: color.iter().map(|a| a.handle).collect(),
            depth: depth.as_ref().map(|a| a.handle),
        };
        if let Some(fb) = self.framebuffer_cache.get(&hash_obj).cloned() {
            return Ok(fb);
        }
        let res = color
            .first()
            .ok_or("no color attachments provided".to_string())?
            .image_holder
            .as_ref()
            .res;
        let res = (res.0, res.1);
        for ca in &color[1..] {
            let ca_res = ca.image_holder.as_ref().res;
            let ca_res = (ca_res.0, ca_res.1);
            if ca_res != res {
                return Err("not all attachments are of same res".to_string());
            }
        }
        if let Some(da) = &depth {
            let da_res = da.image_holder.as_ref().res;
            let da_res = (da_res.0, da_res.1);
            if da_res != res {
                return Err("not all attachments are of same res".to_string());
            }
        }
        let mut attachments: Vec<_> = color.iter().map(|a| a.handle).collect();
        if let Some(da) = &depth {
            attachments.push(da.handle);
        }
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(self.render_pass)
            .attachment_count(attachments.len() as _)
            .attachments(&attachments)
            .width(res.0)
            .height(res.1)
            .layers(1);
        let fb = unsafe {
            self.device_dropper
                .device
                .create_framebuffer(&create_info, None)
                .map_err(|e| format!("vk framebuffer create failed: {e}"))?
        };
        if self.framebuffer_cache.len() >= MAX_CACHED_FBS {
            for (_, fb) in self.framebuffer_cache.drain() {
                unsafe {
                    self.device_dropper.device.destroy_framebuffer(fb, None);
                }
            }
        }
        self.framebuffer_cache.insert(hash_obj, fb);
        Ok(fb)
    }
}

impl rhi2::graphics_pipeline::GraphicsPipeline for GraphicsPipeline {
    type BType = Buffer;

    type IType = Image;

    type IVType = ImageView;

    type SetType = ShaderSet;

    fn set_count(&self) -> usize {
        self.descriptor_gens.len()
    }

    fn pc_size(&self) -> usize {
        self.pc_size
    }

    fn new_set(
        &mut self,
        set_id: usize,
    ) -> Result<Self::SetType, rhi2::graphics_pipeline::GraphicsPipelineErr> {
        self.descriptor_gens[set_id]
            .new_shader_set()
            .map_err(rhi2::graphics_pipeline::GraphicsPipelineErr::SetCreateFailed)
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            for (_, fb) in self.framebuffer_cache.drain() {
                self.device_dropper.device.destroy_framebuffer(fb, None);
            }
        }
    }
}
