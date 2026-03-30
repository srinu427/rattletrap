use std::usize;

use enumflags2::BitFlags;

use crate::{
    HostAccess,
    buffer::{Buffer, BufferFlags},
    command::CommandRecorder,
    graphics_pipeline::{FragmentStageInfo, GraphicsPipeline, VertexStageInfo},
    image::{Format, Image, ImageFlags, ImageView, Sampler},
    shader::{ShaderSet, ShaderSetInfo},
    swapchain::Swapchain,
    sync::TaskFuture,
};

#[derive(Debug, Clone, thiserror::Error)]
pub enum DeviceErr {
    #[error("refreshing resolution failed: {0}")]
    RefreshResError(String),
    #[error("buffer creation failed: {0}")]
    BufferCreateFailed(String),
    #[error("image creation failed: {0}")]
    ImageCreateFailed(String),
    #[error("sampler creation failed: {0}")]
    SamplerCreateFailed(String),
    #[error("graphics pipeline creation failed: {0}")]
    GraphicsPipelineCreateFailed(String),
    #[error("cmd recorder creation failed: {0}")]
    CmdRecorderCreateFailed(String),
    #[error("running commands failed: {0}")]
    RunCmdsFailed(String),
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub dvram: u64,
    pub is_dedicated: bool,
}

pub trait Device {
    type B: Buffer;
    type I: Image;
    type IV: ImageView<I = Self::I>;
    type S: Sampler;
    type SC: Swapchain<I = Self::I, IV = Self::IV, CR = Self::CR, TF = Self::TF>;
    type SS: ShaderSet<B = Self::B, I = Self::I, IV = Self::IV, S = Self::S>;
    type GP: GraphicsPipeline<B = Self::B, I = Self::I, IV = Self::IV, SS = Self::SS>;
    type TF: TaskFuture;
    type CR: CommandRecorder<
            B = Self::B,
            I = Self::I,
            IV = Self::IV,
            GP = Self::GP,
            SS = Self::SS,
            TF = Self::TF,
        >;

    fn gpu_info(&self) -> GpuInfo;
    fn swapchain(&self) -> &Self::SC;
    fn swapchain_mut(&mut self) -> &mut Self::SC;
    fn new_buffer(
        &self,
        size: usize,
        flags: BitFlags<BufferFlags>,
        host_access: HostAccess,
    ) -> Result<Self::B, DeviceErr>;
    fn new_image(
        &self,
        format: Format,
        res: (u32, u32, u32),
        layers: u32,
        flags: BitFlags<ImageFlags>,
        host_access: HostAccess,
    ) -> Result<Self::I, DeviceErr>;
    fn new_sampler(&self) -> Result<Self::S, DeviceErr>;
    fn new_graphics_pipeline(
        &self,
        shader: &str,
        sets: Vec<Vec<ShaderSetInfo>>,
        pc_size: usize,
        vert_stage_info: VertexStageInfo,
        frag_stage_info: FragmentStageInfo,
    ) -> Result<Self::GP, DeviceErr>;
    fn new_cmd_recorder(&self) -> Result<Self::CR, DeviceErr>;
    fn run_work_graph(&self) -> Result<Self::TF, DeviceErr>;
    fn wait_idle(&self);
}
