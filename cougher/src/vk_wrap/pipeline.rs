use std::sync::Arc;

use ash::vk;

use crate::vk_wrap::device::Device;

pub struct RenderPass {
    pub(crate) rp: vk::RenderPass,
    pub(crate) device: Arc<Device>,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_render_pass(self.rp, None);
        }
    }
}

pub struct Pipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) device: Arc<Device>,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DSetBindingInfo {
    UniformBuffer(usize),
    StorageBuffer(usize),
    Sampler2d(usize),
}

impl DSetBindingInfo {
    pub fn vk_type(&self) -> vk::DescriptorType {
        match self {
            DSetBindingInfo::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            DSetBindingInfo::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            DSetBindingInfo::Sampler2d(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }

    pub fn count(&self) -> usize {
        match self {
            DSetBindingInfo::UniformBuffer(c) => *c,
            DSetBindingInfo::StorageBuffer(c) => *c,
            DSetBindingInfo::Sampler2d(c) => *c,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Error creating Vulkan Descriptor Set Layout: {0}")]
    DslCreateError(vk::Result),
    #[error("Error creating Vulkan Pipeline Layout: {0}")]
    PipelineLayoutCreateError(vk::Result),
    #[error("Error creating Vulkan RenderPass: {0}")]
    RenderPassCreateError(vk::Result),
    #[error("Error creating Vulkan Shader Module: {0}")]
    ShaderModCreateError(vk::Result),
    #[error("Error creating Vulkan Pipeline: {0}")]
    PipelineCreateError(vk::Result),
}

pub struct Dsl {
    pub(crate) dsl: vk::DescriptorSetLayout,
    pub(crate) device: Arc<Device>,
}

impl Dsl {
    pub fn new(
        device: &Arc<Device>,
        dynamic: bool,
        bindings: &[DSetBindingInfo],
    ) -> Result<Self, PipelineError> {
        let flags = if dynamic {
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };
        let bindings_vk: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_type(b.vk_type())
                    .descriptor_count(b.count() as _)
            })
            .collect();

        let dsl = unsafe {
            device
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(flags)
                        .bindings(&bindings_vk),
                    None,
                )
                .map_err(PipelineError::DslCreateError)?
        };
        Ok(Self {
            dsl,
            device: device.clone(),
        })
    }
}

impl Drop for Dsl {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}

pub struct PipelineLayout {
    pub(crate) pl: vk::PipelineLayout,
    pub(crate) device: Arc<Device>,
}

impl PipelineLayout {
    pub fn new(device: &Arc<Device>, dsls: &[&Dsl], pc_size: u32) -> Result<Self, PipelineError> {
        let dsls_vk: Vec<_> = dsls.iter().map(|d| d.dsl).collect();
        let mut create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls_vk);
        let pc_range = [vk::PushConstantRange::default()
            .offset(0)
            .size(pc_size)
            .stage_flags(vk::ShaderStageFlags::ALL)];
        if pc_size != 0 {
            create_info = create_info.push_constant_ranges(&pc_range)
        }
        let pl = unsafe {
            device
                .device
                .create_pipeline_layout(&create_info, None)
                .map_err(PipelineError::PipelineLayoutCreateError)?
        };
        Ok(PipelineLayout {
            pl,
            device: device.clone(),
        })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline_layout(self.pl, None);
        }
    }
}

pub struct ShaderModule {
    pub(crate) sm: vk::ShaderModule,
    pub(crate) device: Arc<Device>,
}

impl ShaderModule {
    pub fn new(device: &Arc<Device>, code: &[u8]) -> Result<Self, PipelineError> {
        let shader = unsafe {
            device
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(code.align_to::<u32>().1),
                    None,
                )
                .map_err(PipelineError::ShaderModCreateError)?
        };
        Ok(Self {
            sm: shader,
            device: device.clone(),
        })
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_shader_module(self.sm, None);
        }
    }
}
