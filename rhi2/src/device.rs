use std::usize;

use enumflags2::BitFlags;

use crate::{
    HostAccess,
    buffer::{Buffer, BufferFlags},
    command::CommandRecorder,
    graphics_pipeline::{FragmentStageInfo, GraphicsAttach, GraphicsPipeline, VertexStageInfo},
    image::{Format, Image, ImageFlags, ImageView},
    shader::{ShaderSet, ShaderSetInfo},
    swapchain::Swapchain,
    sync::CpuFuture,
};

#[derive(Debug, Clone, thiserror::Error)]
pub enum DeviceErr {
    #[error("refreshing resolution failed: {0}")]
    RefreshResError(String),
    #[error("buffer creation failed: {0}")]
    BufferCreateFailed(String),
    #[error("image creation failed: {0}")]
    ImageCreateFailed(String),
    #[error("graphics pipeline creation failed: {0}")]
    GraphicsPipelineCreateFailed(String),
    #[error("cmd recorder creation failed: {0}")]
    CmdRecorderCreateFailed(String),
    #[error("running commands failed: {0}")]
    RunCmdsFailed(String),
}

pub trait Device {
    type SC: Swapchain;
    type BType: Buffer;
    type IType: Image;
    type IVType: ImageView<IType = Self::IType>;
    type SSType: ShaderSet<BType = Self::BType, IType = Self::IType, IVType = Self::IVType>;
    type GAType: GraphicsAttach<IVType = Self::IVType>;
    type GPType: GraphicsPipeline<
            BType = Self::BType,
            IType = Self::IType,
            IVType = Self::IVType,
            SetType = Self::SSType,
            AttachType = Self::GAType,
        >;
    type CRType: CommandRecorder;
    type CFType: CpuFuture;

    fn swapchain(&self) -> &Self::SC;
    fn swapchain_mut(&mut self) -> &mut Self::SC;
    fn new_buffer(
        &self,
        size: usize,
        flags: BitFlags<BufferFlags>,
        host_access: HostAccess,
    ) -> Result<Self::BType, DeviceErr>;
    fn new_image(
        &self,
        format: Format,
        res: (u32, u32, u32),
        layers: u32,
        flags: BitFlags<ImageFlags>,
        host_access: HostAccess,
    ) -> Result<Self::IType, DeviceErr>;
    fn new_graphics_pipeline(
        &self,
        shader: &str,
        sets: Vec<Vec<ShaderSetInfo>>,
        pc_size: usize,
        vert_stage_info: VertexStageInfo,
        frag_stage_info: FragmentStageInfo,
    ) -> Result<Self::GPType, DeviceErr>;
    fn new_cmd_recorders(&self, count: usize) -> Result<Vec<Self::CRType>, DeviceErr>;
}
