use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(Debug, thiserror::Error)]
pub enum DescriptorSetLayoutError {
    #[error("Descriptor set layout creation error: {0}")]
    CreateError(vk::Result),
}

#[derive(Debug, Clone, Copy)]
pub struct DescriptorSetLayoutBinding {
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub partially_bound: bool,
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct DescriptorSetLayout {
    #[get_copy = "pub"]
    layout: vk::DescriptorSetLayout,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl DescriptorSetLayout {
    pub fn new(
        device: Arc<LogicalDevice>,
        bindings: &[(vk::DescriptorType, u32, bool)],
    ) -> Result<Self, DescriptorSetLayoutError> {
        let bindless_binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;

        let bindless_layout_flags = vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;

        let vk_bindings = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(b.0)
                    .descriptor_count(b.1)
                    .stage_flags(vk::ShaderStageFlags::ALL)
            })
            .collect::<Vec<_>>();

        let binding_flags = bindings
            .iter()
            .map(|b| if b.2 { bindless_binding_flags } else { vk::DescriptorBindingFlags::empty() })
            .collect::<Vec<_>>();
        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&binding_flags);

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&vk_bindings)
            .flags(bindless_layout_flags)
            .push_next(&mut binding_flags_info);
        let layout = unsafe {
            device
                .device()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(DescriptorSetLayoutError::CreateError)?
        };
        Ok(Self { layout, device })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
