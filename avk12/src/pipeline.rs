use std::{
    ffi::CString,
    fs,
    sync::{Arc, Mutex, PoisonError, RwLock},
};

use anyhow::Context;
use ash::vk;
use hashbrown::HashMap;
use typed_builder::TypedBuilder;

use crate::{
    device::DeviceDropper,
    resource::{BufferRef, ImageView, Sampler},
};

static MAX_FB_CACHE: usize = 128;

pub fn safe_str_to_cstring(str: String) -> CString {
    let mut msg = str.into_bytes();
    let cstr = loop {
        match CString::new(msg) {
            Ok(cstr) => break cstr,
            Err(err) => {
                let idx = err.nul_position();
                msg = err.into_vec();
                msg.remove(idx);
            }
        }
    };
    cstr
}

#[derive(Debug, Clone)]
pub enum BindInfo {
    UniformBuffer(usize),
    StorageBuffer(usize),
    Sampler(usize),
    Texture(usize),
    Sampler2D(usize),
}

impl BindInfo {
    fn to_vk(&self) -> vk::DescriptorType {
        match self {
            BindInfo::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            BindInfo::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            BindInfo::Sampler(_) => vk::DescriptorType::SAMPLER,
            BindInfo::Texture(_) => vk::DescriptorType::SAMPLED_IMAGE,
            BindInfo::Sampler2D(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }
}

struct DPoolDropper {
    handle: vk::DescriptorPool,
    dd: Arc<DeviceDropper>,
}

impl Drop for DPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.dd.device.destroy_descriptor_pool(self.handle, None);
        }
    }
}

pub(crate) struct DPool {
    layout: Vec<BindInfo>,
    vk_layout: vk::DescriptorSetLayout,
    dropper: Arc<DPoolDropper>,
}

impl DPool {
    fn pool_sizes(layout: &Vec<BindInfo>) -> Vec<vk::DescriptorPoolSize> {
        let mut uni_buf_sizes =
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER);
        let mut sto_buf_sizes =
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER);
        let mut samp_sizes = vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER);
        let mut tex_sizes = vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE);
        let mut samp2_sizes =
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        for b_info in layout {
            match b_info {
                BindInfo::UniformBuffer(c) => {
                    uni_buf_sizes.descriptor_count = *c as _;
                }
                BindInfo::StorageBuffer(c) => {
                    sto_buf_sizes.descriptor_count = *c as _;
                }
                BindInfo::Sampler(c) => {
                    samp_sizes.descriptor_count = *c as _;
                }
                BindInfo::Texture(c) => {
                    tex_sizes.descriptor_count = *c as _;
                }
                BindInfo::Sampler2D(c) => {
                    samp2_sizes.descriptor_count = *c as _;
                }
            }
        }
        let mut out = vec![];
        if uni_buf_sizes.descriptor_count > 0 {
            out.push(uni_buf_sizes);
        }
        if sto_buf_sizes.descriptor_count > 0 {
            out.push(sto_buf_sizes);
        }
        if samp_sizes.descriptor_count > 0 {
            out.push(samp_sizes);
        }
        if tex_sizes.descriptor_count > 0 {
            out.push(tex_sizes);
        }
        if samp2_sizes.descriptor_count > 0 {
            out.push(samp2_sizes);
        }
        out
    }

    pub fn new(dd: &Arc<DeviceDropper>, layout: Vec<BindInfo>) -> anyhow::Result<Self> {
        let bindings: Vec<_> = layout
            .iter()
            .enumerate()
            .map(|(i, b)| match b {
                BindInfo::UniformBuffer(c) => vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(*c as _),
                BindInfo::StorageBuffer(c) => vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(*c as _),
                BindInfo::Sampler(c) => vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(*c as _),
                BindInfo::Texture(c) => vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(*c as _),
                BindInfo::Sampler2D(c) => vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(*c as _),
            })
            .collect();
        let dsl = unsafe {
            dd.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .context("create dsl failed")?
        };
        let pool_sizes = Self::pool_sizes(&layout);
        let dpool = unsafe {
            dd.device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(128)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .context("create dpool failed")?
        };
        Ok(Self {
            layout,
            vk_layout: dsl,
            dropper: Arc::new(DPoolDropper {
                handle: dpool,
                dd: dd.clone(),
            }),
        })
    }

    pub fn new_set(&mut self) -> anyhow::Result<DSet> {
        let set_alloc_res = unsafe {
            self.dropper.dd.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.dropper.handle)
                    .set_layouts(&[self.vk_layout]),
            )
        };
        let mut captures = Vec::with_capacity(self.layout.len());
        for b_type in &self.layout {
            match b_type {
                BindInfo::UniformBuffer(c) => captures.push(DSetData::Buffers(vec![None; *c])),
                BindInfo::StorageBuffer(c) => captures.push(DSetData::Buffers(vec![None; *c])),
                BindInfo::Sampler(c) => captures.push(DSetData::Samplers(vec![None; *c])),
                BindInfo::Texture(c) => captures.push(DSetData::ImageViews(vec![None; *c])),
                BindInfo::Sampler2D(c) => captures.push(DSetData::Sampler2Ds(vec![None; *c])),
            }
        }

        match set_alloc_res {
            Ok(dset) => Ok(DSet {
                captures,
                layout: self.layout.clone(),
                set: dset[0],
                pool_dropper: self.dropper.clone(),
            }),
            Err(e) => match e {
                vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                    let pool_sizes = Self::pool_sizes(&self.layout);
                    let dpool = unsafe {
                        self.dropper
                            .dd
                            .device
                            .create_descriptor_pool(
                                &vk::DescriptorPoolCreateInfo::default()
                                    .max_sets(128)
                                    .pool_sizes(&pool_sizes),
                                None,
                            )
                            .context("create dpool failed")?
                    };
                    self.dropper = Arc::new(DPoolDropper {
                        handle: dpool,
                        dd: self.dropper.dd.clone(),
                    });
                    let dset = unsafe {
                        self.dropper
                            .dd
                            .device
                            .allocate_descriptor_sets(
                                &vk::DescriptorSetAllocateInfo::default()
                                    .descriptor_pool(self.dropper.handle)
                                    .set_layouts(&[self.vk_layout]),
                            )
                            .context("allocate dset failed")?
                    };
                    Ok(DSet {
                        captures,
                        layout: self.layout.clone(),
                        set: dset[0],
                        pool_dropper: self.dropper.clone(),
                    })
                }
                _ => Err(anyhow::Error::new(e).context("allocate dset failed")),
            },
        }
    }
}

impl Drop for DPool {
    fn drop(&mut self) {
        unsafe {
            self.dropper
                .dd
                .device
                .destroy_descriptor_set_layout(self.vk_layout, None);
        }
    }
}

#[derive(Clone)]
pub struct CombinedImageSampler {
    pub(crate) view: ImageView,
    pub(crate) sampler: Sampler,
}

pub enum DSetWriteData<'a> {
    Buffers(Vec<&'a BufferRef>),
    Samplers(Vec<&'a Sampler>),
    Textures(Vec<&'a ImageView>),
    Sampler2Ds(Vec<CombinedImageSampler>),
}

pub(crate) enum DSetData {
    Buffers(Vec<Option<BufferRef>>),
    Samplers(Vec<Option<Sampler>>),
    ImageViews(Vec<Option<ImageView>>),
    Sampler2Ds(Vec<Option<CombinedImageSampler>>),
}

pub struct DSet {
    pub(crate) set: vk::DescriptorSet,
    pub(crate) captures: Vec<DSetData>,
    layout: Vec<BindInfo>,
    pool_dropper: Arc<DPoolDropper>,
}

impl DSet {
    pub fn update_binding(
        &mut self,
        binding: usize,
        offset: usize,
        data: DSetWriteData,
    ) -> anyhow::Result<()> {
        let capture_mut = match self.captures.get_mut(binding) {
            Some(c) => c,
            None => {
                return Err(anyhow::Error::msg(format!(
                    "input binding ({}) out of range. dset binding count: {}",
                    binding,
                    self.captures.len(),
                )));
            }
        };
        match (data, capture_mut) {
            (DSetWriteData::Buffers(new_bufs), DSetData::Buffers(old_bufs)) => {
                let dst_bufs = &mut old_bufs[offset..];
                let copy_size = dst_bufs.len().min(dst_bufs.len());
                for i in 0..copy_size {
                    dst_bufs[i] = Some(new_bufs[i].clone());
                }
                if copy_size != 0 {
                    unsafe {
                        self.pool_dropper.dd.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::default()
                                .dst_set(self.set)
                                .descriptor_type(self.layout[binding].to_vk())
                                .descriptor_count(copy_size as _)
                                .dst_binding(binding as _)
                                .dst_array_element(offset as _)
                                .buffer_info(
                                    &(0..copy_size)
                                        .map(|i| {
                                            vk::DescriptorBufferInfo::default()
                                                .buffer(new_bufs[i].dropper.handle)
                                                .range(vk::WHOLE_SIZE)
                                        })
                                        .collect::<Vec<_>>(),
                                )],
                            &[],
                        );
                    }
                }
            }
            (DSetWriteData::Samplers(new_samps), DSetData::Samplers(old_samps)) => {
                let dst_samps = &mut old_samps[offset..];
                let copy_size = dst_samps.len().min(dst_samps.len());
                for i in 0..copy_size {
                    dst_samps[i] = Some(new_samps[i].clone());
                }
                if copy_size != 0 {
                    unsafe {
                        self.pool_dropper.dd.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::default()
                                .dst_set(self.set)
                                .descriptor_type(self.layout[binding].to_vk())
                                .descriptor_count(copy_size as _)
                                .dst_binding(binding as _)
                                .dst_array_element(offset as _)
                                .image_info(
                                    &(0..copy_size)
                                        .map(|i| {
                                            vk::DescriptorImageInfo::default()
                                                .sampler(new_samps[i].dropper.handle)
                                        })
                                        .collect::<Vec<_>>(),
                                )],
                            &[],
                        );
                    }
                }
            }
            (DSetWriteData::Textures(new_texs), DSetData::ImageViews(old_texs)) => {
                let dst_texs = &mut old_texs[offset..];
                let copy_size = dst_texs.len().min(dst_texs.len());
                for i in 0..copy_size {
                    dst_texs[i] = Some(new_texs[i].clone());
                }
                if copy_size != 0 {
                    unsafe {
                        self.pool_dropper.dd.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::default()
                                .dst_set(self.set)
                                .descriptor_type(self.layout[binding].to_vk())
                                .descriptor_count(copy_size as _)
                                .dst_binding(binding as _)
                                .dst_array_element(offset as _)
                                .image_info(
                                    &(0..copy_size)
                                        .map(|i| {
                                            vk::DescriptorImageInfo::default()
                                                .image_view(new_texs[i].handle)
                                                .image_layout(
                                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                                )
                                        })
                                        .collect::<Vec<_>>(),
                                )],
                            &[],
                        );
                    }
                }
            }
            (DSetWriteData::Sampler2Ds(new_texs), DSetData::Sampler2Ds(old_texs)) => {
                let dst_texs = &mut old_texs[offset..];
                let copy_size = dst_texs.len().min(dst_texs.len());
                for i in 0..copy_size {
                    dst_texs[i] = Some(new_texs[i].clone());
                }
                if copy_size != 0 {
                    unsafe {
                        self.pool_dropper.dd.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::default()
                                .dst_set(self.set)
                                .descriptor_type(self.layout[binding].to_vk())
                                .descriptor_count(copy_size as _)
                                .dst_binding(binding as _)
                                .dst_array_element(offset as _)
                                .image_info(
                                    &(0..copy_size)
                                        .map(|i| {
                                            vk::DescriptorImageInfo::default()
                                                .image_view(new_texs[i].view.handle)
                                                .image_layout(
                                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                                )
                                                .sampler(new_texs[i].sampler.dropper.handle)
                                        })
                                        .collect::<Vec<_>>(),
                                )],
                            &[],
                        );
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

struct ShaderModDropper {
    handle: vk::ShaderModule,
    dd: Arc<DeviceDropper>,
}

impl ShaderModDropper {
    fn new(dd: &Arc<DeviceDropper>, code: &[u32]) -> anyhow::Result<Self> {
        let handle = unsafe {
            dd.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
                .context("create shader mod failed")?
        };
        Ok(Self {
            handle,
            dd: dd.clone(),
        })
    }
}

impl Drop for ShaderModDropper {
    fn drop(&mut self) {
        unsafe {
            self.dd.device.destroy_shader_module(self.handle, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RasterMode {
    Fill,
    Edge(f32),
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct RasterConfig {
    #[builder(default=RasterMode::Fill)]
    pub mode: RasterMode,
    #[builder(default = true)]
    pub draw_front: bool,
    #[builder(default = false)]
    pub draw_back: bool,
    #[builder(default = false)]
    pub clockwise: bool,
}

impl RasterConfig {
    fn cull_mode_vk(&self) -> vk::CullModeFlags {
        let mut out = vk::CullModeFlags::empty();
        if !self.draw_back {
            out |= vk::CullModeFlags::BACK;
        }
        if !self.draw_front {
            out |= vk::CullModeFlags::FRONT;
        }
        out
    }

    fn front_face_vk(&self) -> vk::FrontFace {
        if self.clockwise {
            vk::FrontFace::CLOCKWISE
        } else {
            vk::FrontFace::COUNTER_CLOCKWISE
        }
    }

    fn polygon_mode_vk(&self) -> vk::PolygonMode {
        match self.mode {
            RasterMode::Fill => vk::PolygonMode::FILL,
            RasterMode::Edge(_) => vk::PolygonMode::LINE,
        }
    }

    fn line_width_vk(&self) -> f32 {
        match self.mode {
            RasterMode::Fill => 1.0,
            RasterMode::Edge(w) => w,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VertexAttribute {
    Vec3,
    Vec4,
}

impl VertexAttribute {
    pub fn size(&self) -> u32 {
        match self {
            VertexAttribute::Vec3 => 3 * 4,
            VertexAttribute::Vec4 => 4 * 4,
        }
    }

    pub(crate) fn to_vk(&self) -> vk::Format {
        match self {
            VertexAttribute::Vec3 => vk::Format::R32G32B32_SFLOAT,
            VertexAttribute::Vec4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttachInfo {
    pub format: vk::Format,
    pub clear: bool,
    pub store: bool,
}

impl AttachInfo {
    pub(crate) fn load_op(&self) -> vk::AttachmentLoadOp {
        if self.clear {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        }
    }

    pub(crate) fn store_op(&self) -> vk::AttachmentStoreOp {
        if self.store {
            vk::AttachmentStoreOp::STORE
        } else {
            vk::AttachmentStoreOp::DONT_CARE
        }
    }
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct VertexConfig {
    pub shader: String,
    #[builder(default="vs_main".to_string())]
    pub fn_name: String,
    #[builder(default)]
    pub attribs: Vec<VertexAttribute>,
    #[builder(default)]
    pub stride: usize,
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct FragmentConfig {
    pub shader: String,
    #[builder(default="fs_main".to_string())]
    pub fn_name: String,
    #[builder(default)]
    pub attachments: Vec<AttachInfo>,
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct GraphicsPipelineCreateInfo {
    pub set_layouts: Vec<Vec<BindInfo>>,
    pub vert_conf: VertexConfig,
    pub frag_conf: FragmentConfig,
    #[builder(default = 0)]
    pub pc_size: usize,
    #[builder(default = RasterConfig::builder().build())]
    pub raster_conf: RasterConfig,
    #[builder(default)]
    pub depth_conf: Option<AttachInfo>,
}

pub struct GraphicsPipelineDropper {
    pub(crate) render_pass: vk::RenderPass,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) dpools: Mutex<Vec<DPool>>,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    framebuffer_cache: RwLock<HashMap<Vec<vk::ImageView>, vk::Framebuffer>>,
    dd: Arc<DeviceDropper>,
}

impl Drop for GraphicsPipelineDropper {
    fn drop(&mut self) {
        for (_, fb) in self
            .framebuffer_cache
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
        {
            unsafe {
                self.dd.device.destroy_framebuffer(fb, None);
            }
        }
        unsafe {
            self.dd
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.dd.device.destroy_pipeline(self.pipeline, None);
            self.dd.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

pub struct GraphicsPipeline {
    pub info: GraphicsPipelineCreateInfo,
    pub(crate) dropper: Arc<GraphicsPipelineDropper>,
}

impl GraphicsPipeline {
    fn make_render_pass(
        dd: &Arc<DeviceDropper>,
        info: &GraphicsPipelineCreateInfo,
    ) -> anyhow::Result<vk::RenderPass> {
        let mut attachments = vec![];
        let mut color_refs = vec![];
        let mut depth_ref = None;
        for a in &info.frag_conf.attachments {
            color_refs.push(
                vk::AttachmentReference::default()
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .attachment(attachments.len() as _),
            );
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(a.format)
                    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(a.load_op())
                    .store_op(a.store_op())
                    .samples(vk::SampleCountFlags::TYPE_1),
            )
        }
        if let Some(a) = &info.depth_conf {
            depth_ref = Some(
                vk::AttachmentReference::default()
                    .layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .attachment(attachments.len() as _),
            );
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(a.format)
                    .initial_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(a.load_op())
                    .store_op(a.store_op())
                    .samples(vk::SampleCountFlags::TYPE_1),
            )
        }
        let mut subpass_info = vk::SubpassDescription::default().color_attachments(&color_refs);
        if let Some(dr) = &depth_ref {
            subpass_info = subpass_info.depth_stencil_attachment(dr);
        }
        let rp_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(core::slice::from_ref(&subpass_info));
        let render_pass = unsafe {
            dd.device
                .create_render_pass(&rp_create_info, None)
                .context("create renderpass failed")?
        };
        Ok(render_pass)
    }

    fn _load_wgsl(
        path: &str,
        vs_main: &str,
        fs_main: &str,
    ) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
        let wgsl_data = fs::read_to_string(path).context(format!("IO error at {path}"))?;
        let naga_ir = naga::front::wgsl::parse_str(&wgsl_data).context("WGSL parse failed")?;
        let mod_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&naga_ir)
        .context("WGSL validation failed")?;
        let vert_code = naga::back::spv::write_vec(
            &naga_ir,
            &mod_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Vertex,
                entry_point: vs_main.to_string(),
            }),
        )
        .context(format!(
            "extracting vertex shader with entry point {vs_main} failed"
        ))?;
        let frag_code = naga::back::spv::write_vec(
            &naga_ir,
            &mod_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Fragment,
                entry_point: fs_main.to_string(),
            }),
        )
        .context(format!(
            "extracting fragment shader with entry point {fs_main} failed"
        ))?;
        Ok((vert_code, frag_code))
    }

    fn load_glsl(
        path: &str,
        entry_point: &str,
        stage: naga::ShaderStage,
    ) -> anyhow::Result<Vec<u32>> {
        let glsl_data = fs::read_to_string(path).context(format!("IO error at {path}"))?;
        let mut naga_fe = naga::front::glsl::Frontend::default();
        let options = naga::front::glsl::Options::from(stage);
        let naga_ir = naga_fe
            .parse(&options, &glsl_data)
            .context("parsing GLSL failed")?;
        let mod_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&naga_ir)
        .context("WGSL validation failed")?;
        let spv_code = naga::back::spv::write_vec(
            &naga_ir,
            &mod_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: stage,
                entry_point: entry_point.to_string(),
            }),
        )
        .context(format!(
            "extracting shader with entry point {entry_point} failed"
        ))?;
        Ok(spv_code)
    }

    pub(crate) fn new(
        dd: &Arc<DeviceDropper>,
        info: GraphicsPipelineCreateInfo,
    ) -> anyhow::Result<Self> {
        let vs_code = Self::load_glsl(
            &info.vert_conf.shader,
            &info.vert_conf.fn_name,
            naga::ShaderStage::Vertex,
        )
        .context("vertex shader compile failed")?;
        let fs_code = Self::load_glsl(
            &info.frag_conf.shader,
            &info.frag_conf.fn_name,
            naga::ShaderStage::Fragment,
        )
        .context("fragment shader compile failed")?;
        let vs_mod = ShaderModDropper::new(dd, &vs_code)?;
        let fs_mod = ShaderModDropper::new(dd, &fs_code)?;
        let render_pass = Self::make_render_pass(dd, &info)?;
        let dpools: Vec<_> = info
            .set_layouts
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, sl)| DPool::new(dd, sl).context(format!("set #{i}")))
            .collect::<Result<_, _>>()?;
        let vk_set_layouts: Vec<_> = dpools.iter().map(|d| d.vk_layout).collect();
        let pc_range = vk::PushConstantRange::default()
            .size(info.pc_size as _)
            .stage_flags(vk::ShaderStageFlags::ALL);
        let mut pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&vk_set_layouts);
        if pc_range.size > 0 {
            pipeline_layout_create_info =
                pipeline_layout_create_info.push_constant_ranges(core::slice::from_ref(&pc_range));
        }
        let pipeline_layout = unsafe {
            dd.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .context("pipeline layout creation failed")?
        };
        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let mut vertex_attribute_descriptions = vec![];
        let mut curr_vert_offset = 0;
        for (i, a) in info.vert_conf.attribs.iter().enumerate() {
            let vad = vk::VertexInputAttributeDescription::default()
                .format(a.to_vk())
                .location(i as _)
                .offset(curr_vert_offset);
            curr_vert_offset += a.size();
            vertex_attribute_descriptions.push(vad);
        }
        let vertex_binding_description = vk::VertexInputBindingDescription::default()
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(info.vert_conf.stride as _);
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attribute_descriptions)
            .vertex_binding_descriptions(core::slice::from_ref(&vertex_binding_description));
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(info.raster_conf.cull_mode_vk())
            .front_face(info.raster_conf.front_face_vk())
            .polygon_mode(info.raster_conf.polygon_mode_vk())
            .line_width(info.raster_conf.line_width_vk());
        let vert_name_cstr = safe_str_to_cstring(info.vert_conf.fn_name.clone());
        let frag_name_cstr = safe_str_to_cstring(info.frag_conf.fn_name.clone());
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_mod.handle)
                .name(&vert_name_cstr),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_mod.handle)
                .name(&frag_name_cstr),
        ];
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachments: Vec<_> = (0..info.frag_conf.attachments.len())
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);
        let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .render_pass(render_pass)
            .layout(pipeline_layout)
            .dynamic_state(&dynamic_state)
            .viewport_state(&viewport_state)
            .stages(&stages)
            .input_assembly_state(&input_assembly_state)
            .vertex_input_state(&vertex_input_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blending);
        if let Some(_) = &info.depth_conf {
            depth_state = depth_state
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            graphics_pipeline_create_info =
                graphics_pipeline_create_info.depth_stencil_state(&depth_state);
        }
        let pipeline = unsafe {
            dd.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphics_pipeline_create_info],
                    None,
                )
                .map_err(|e| e.1)
                .context("graphics pipeline creation failed")?[0]
        };
        drop(vs_mod);
        drop(fs_mod);
        Ok(Self {
            info,
            dropper: Arc::new(GraphicsPipelineDropper {
                render_pass,
                pipeline,
                dpools: Mutex::new(dpools),
                pipeline_layout,
                framebuffer_cache: RwLock::new(HashMap::new()),
                dd: dd.clone(),
            }),
        })
    }

    pub(crate) fn get_fb(&mut self, views: &[ImageView]) -> anyhow::Result<vk::Framebuffer> {
        if views.len() == 0 {
            return Err(anyhow::Error::msg("no input views given"));
        }
        // Validate if views are of same size
        let width = views[0].image_droppper.info.res.0;
        let height = views[0].image_droppper.info.res.1;
        for view in &views[1..] {
            if width != view.image_droppper.info.res.0 || height != view.image_droppper.info.res.1 {
                return Err(anyhow::Error::msg(
                    "image views have a mismatch in resolution",
                ));
            }
        }
        let view_handles: Vec<_> = views.iter().map(|v| v.handle).collect();
        let ex_fb = self
            .dropper
            .framebuffer_cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&view_handles)
            .cloned();
        match ex_fb {
            Some(fb) => Ok(fb),
            None => {
                let fb = unsafe {
                    self.dropper
                        .dd
                        .device
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::default()
                                .render_pass(self.dropper.render_pass)
                                .width(views[0].image_droppper.info.res.0)
                                .height(views[0].image_droppper.info.res.1)
                                .layers(1)
                                .attachments(&view_handles),
                            None,
                        )
                        .context("creating vk framebuffer failed")?
                };
                let fb_count = self
                    .dropper
                    .framebuffer_cache
                    .read()
                    .unwrap_or_else(PoisonError::into_inner)
                    .len();
                if fb_count >= MAX_FB_CACHE {
                    for (_, fb) in self
                        .dropper
                        .framebuffer_cache
                        .write()
                        .unwrap_or_else(PoisonError::into_inner)
                        .drain()
                    {
                        unsafe {
                            self.dropper.dd.device.destroy_framebuffer(fb, None);
                        }
                    }
                }
                self.dropper
                    .framebuffer_cache
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(view_handles, fb);
                Ok(fb)
            }
        }
    }

    pub fn new_set(&self, set_id: usize) -> anyhow::Result<DSet> {
        let mut dpool_mut = self
            .dropper
            .dpools
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match dpool_mut.get_mut(set_id) {
            Some(dpool) => dpool.new_set(),
            None => {
                return Err(anyhow::Error::msg(format!(
                    "set {set_id} out of range, set count: {}",
                    dpool_mut.len()
                )));
            }
        }
    }
}
