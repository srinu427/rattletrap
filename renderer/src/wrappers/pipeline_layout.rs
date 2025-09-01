use std::sync::Arc;

use ash::vk;

use crate::wrappers::{descriptor_set_layout::DescriptorSetLayout, logical_device::LogicalDevice};

#[derive(Debug, thiserror::Error)]
pub enum PipelineLayoutError {
    #[error("Pipeline layout creation error: {0}")]
    CreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct PipelineLayout {
    #[get_copy = "pub"]
    pipeline_layout: vk::PipelineLayout,
    #[get = "pub"]
    set_layouts: Vec<Arc<DescriptorSetLayout>>,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl PipelineLayout {
    pub fn new(
        device: Arc<LogicalDevice>,
        set_layouts: Vec<Arc<DescriptorSetLayout>>,
    ) -> Result<Self, PipelineLayoutError> {
        let vk_set_layouts = set_layouts
            .iter()
            .map(|layout| layout.layout())
            .collect::<Vec<_>>();
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&vk_set_layouts);

        let pipeline_layout = unsafe {
            device
                .device()
                .create_pipeline_layout(&create_info, None)
                .map_err(PipelineLayoutError::CreateError)?
        };

        Ok(Self {
            pipeline_layout,
            set_layouts,
            device,
        })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
