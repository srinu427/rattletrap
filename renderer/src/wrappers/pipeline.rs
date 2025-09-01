use std::sync::Arc;

use ash::vk;

use crate::wrappers::{pipeline_layout::PipelineLayout, render_pass::RenderPass};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    // No errors for now
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Pipeline {
    #[get_copy = "pub"]
    pipeline: vk::Pipeline,
    #[get = "pub"]
    layout: Arc<PipelineLayout>,
    #[get = "pub"]
    render_pass: Arc<RenderPass>,
}

impl Pipeline {
    pub fn new(
        render_pass: Arc<RenderPass>,
        layout: Arc<PipelineLayout>,
        pipeline: vk::Pipeline,
    ) -> Self {
        Self {
            render_pass,
            pipeline,
            layout,
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.layout
                .device()
                .device()
                .destroy_pipeline(self.pipeline, None);
        }
    }
}
