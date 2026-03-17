use crate::{
    Capped,
    buffer::Buffer,
    graphics_pipeline::{GraphicsAttach, GraphicsPipeline},
    image::Image,
    shader::ShaderSet,
    sync::{PipelineStage, TaskFuture},
};

#[derive(Debug, thiserror::Error)]
pub enum CommandErr {
    #[error("future creation error: {0}")]
    RunErr(String),
}

pub trait GraphicsCommandRecorder {
    type B: Buffer;
    type SS: ShaderSet;

    fn bind_vbs(&mut self, vbs: &[Self::B]);
    fn bind_ib(&mut self, ib: &Self::B, is_16bit: bool);
    fn bind_sets(&mut self, sets: &[Self::SS]);
    fn set_pc(&mut self, data: &[u8]);
}

pub trait CommandRecorder: Sized {
    type B: Buffer;
    type I: Image;
    type GP: GraphicsPipeline;
    type GA: GraphicsAttach;
    type SS: ShaderSet;
    type GCR: GraphicsCommandRecorder;
    type TF: TaskFuture;

    fn copy_b2b(&mut self, src: &Self::B, src_offset: usize, dst: &Self::B, dst_offset: usize);
    fn copy_b2i(&mut self, src: &Self::B, dst: &Self::I);
    fn graphics(self, pipeline: &Self::GP, attach: &Self::GA) -> Self::GCR;
    fn finish_graphics(gcr: Self::GCR) -> Self;
    fn blit(&mut self, src: &Self::I, dst: &Self::I);
    fn keep_buffer_alive(&mut self, buffer: Capped<Self::B>);
    fn run(
        self,
        deps: Vec<(Self::TF, PipelineStage, PipelineStage)>,
    ) -> Result<Self::TF, CommandErr>;
}
