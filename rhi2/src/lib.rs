use enumflags2::BitFlags;

pub use enumflags2;

use buffer::Buffer;

use crate::{
    buffer::BufferFlags,
    command::{CmdFuture, Command},
    graphics_pipeline::{FragmentStageInfo, GraphicsAttach, GraphicsPipeline, VertexStageInfo},
    image::{Image, ImageDimension, ImageFlags, ImageView},
    shader::{ShaderSet, ShaderSetInfo},
    swapchain::Swapchain,
};

pub mod buffer;
pub mod command;
pub mod graphics_pipeline;
pub mod image;
pub mod shader;
pub mod swapchain;

#[derive(Debug, Clone, Copy)]
pub enum HostAccess {
    None,
    Read,
    Write,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DeviceErr {
    #[error("buffer creation failed")]
    BufferCreateFailed,
    #[error("image creation failed")]
    ImageCreateFailed,
}

pub trait Device {
    type BType: Buffer;
    type IType: Image;
    type IVType: ImageView<IType = Self::IType>;
    type SCType: Swapchain<IType = Self::IType, IVType = Self::IVType>;
    type SSType: ShaderSet<BType = Self::BType, IType = Self::IType, IVType = Self::IVType>;
    type GAType: GraphicsAttach<IVType = Self::IVType>;
    type GPType: GraphicsPipeline<
            BType = Self::BType,
            IType = Self::IType,
            IVType = Self::IVType,
            SetType = Self::SSType,
            AttachType = Self::IVType,
        >;
    type FType: CmdFuture;

    fn swapchain(&self) -> &Self::SCType;
    fn swapchain_mut(&mut self) -> &mut Self::SCType;
    fn new_buffer(
        &self,
        size: usize,
        flags: BitFlags<BufferFlags>,
        host_access: HostAccess,
    ) -> Result<Self::BType, DeviceErr>;
    fn new_image(
        &self,
        dim: ImageDimension,
        res: (u32, u32, u32),
        flags: BitFlags<ImageFlags>,
    ) -> Result<Self::IType, DeviceErr>;
    fn new_graphics_pipeline(
        &self,
        shader: &str,
        sets: Vec<ShaderSetInfo>,
        pc_size: usize,
        vert_stage_info: VertexStageInfo,
        frag_stage_info: FragmentStageInfo,
    ) -> Self::GPType;
    fn run_commands(
        &self,
        commands: Command<Self::BType, Self::IType, Self::GPType, Self::GAType, Self::SSType>,
    ) -> Self::FType;
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub dvram: u64,
    pub is_dedicated: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InstanceErr {
    #[error("device creation failed")]
    DeviceCreateFailed,
}

pub trait Instance {
    type DType: Device;
    fn get_gpus(&self) -> &Vec<GpuInfo>;
    fn init_device(self, gpu_id: usize) -> Result<Self::DType, InstanceErr>;
}
