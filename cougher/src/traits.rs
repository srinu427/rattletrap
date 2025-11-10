use core::error::Error;
use enumflags2::{BitFlags, bitflags};
use hashbrown::HashMap;
use slotmap::new_key_type;

pub trait ApiLoader {
    type GpuInfoType: GpuInfo;
    type ContextType: GpuContext;
    type E: Error;

    fn list_supported_gpus(&self) -> Vec<Self::GpuInfoType>;
    fn new_gpu_context(self, gpu: Self::GpuInfoType) -> Result<Self::ContextType, Self::E>;
}

pub trait GpuInfo {
    fn name(&self) -> String;
    fn vram(&self) -> u64;
    fn is_dedicated(&self) -> bool;
}

pub trait GpuContext {
    type MP: MemoryPool;
    type SwapchainType: Swapchain<GFutType = Self::SemType, CFutType = Self::FenType>;
    type PSetType: PipelineSet<MP = Self::MP>;
    type PAttachType: GraphicsPassAttachments<MP = Self::MP>;
    type QType: GpuExecutor<GFutType = Self::SemType, CFutType = Self::FenType>;
    type GPassType: GraphicsPass<MP = Self::MP, PSetType = Self::PSetType, PAttachType = Self::PAttachType>;
    type SemType: GpuFuture;
    type FenType: CpuFuture;
    type E: Error
        + From<image::ImageError>
        + From<<Self::GPassType as GraphicsPass>::E>
        + From<<Self::SwapchainType as Swapchain>::E>
        + From<<Self::FenType as CpuFuture>::E>
        + From<<Self::QType as GpuExecutor>::E>;

    fn new_allocator(&self) -> Result<Self::MP, Self::E>;
    fn new_swapchain(&self, usages: BitFlags<ImageUsage>) -> Result<Self::SwapchainType, Self::E>;
    fn new_gpu_future(&self) -> Result<Self::SemType, Self::E>;
    fn new_cpu_future(&self, signaled: bool) -> Result<Self::FenType, Self::E>;
    fn new_graphics_pass(
        &self,
        attachments: Vec<ImageFormat>,
        subpass_infos: Vec<SubpassInfo>,
        max_sets: usize,
    ) -> Result<Self::GPassType, Self::E>;
    fn get_queue(&mut self) -> Result<Self::QType, Self::E>;
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum QueueType {
    Graphics,
}

pub enum GpuCommand<'a, MP: MemoryPool, G: GraphicsPass<MP = MP>> {
    Image2dUsageHint {
        image: Image2dId,
        mp: &'a MP,
        usage: ImageUsage,
    },
    CopyBufferToImage2d {
        src: BufferId,
        src_mp: &'a MP,
        dst: Image2dId,
        dst_mp: &'a MP,
    },
    BlitImage2d {
        src: Image2dId,
        src_mp: &'a MP,
        dst: Image2dId,
        dst_mp: &'a MP,
    },
    RunGraphicsPass {
        pass: &'a G,
        attachments: &'a G::PAttachType,
        commands: Vec<GraphicsPassCommand<'a, G::PSetType>>,
    },
}

pub enum GraphicsPassCommand<'a, PS: PipelineSet> {
    BindSubpass { idx: usize, sets: Vec<&'a PS> },
    Draw(usize),
}

pub trait GpuExecutor {
    type MP: MemoryPool;
    type GPass: GraphicsPass<MP = Self::MP>;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error;

    fn type_(&self) -> QueueType;
    fn new_command_list(&mut self, name: &str) -> Result<(), Self::E>;
    fn update_command_list(
        &mut self,
        name: &str,
        commands: Vec<GpuCommand<Self::MP, Self::GPass>>,
    ) -> Result<(), Self::E>;
    fn run_command_lists(
        &self,
        lists: &[&str],
        wait_for: Vec<&Self::GFutType>,
        emit_gfuts: Vec<&Self::GFutType>,
        emit_cfut: Option<&Self::CFutType>,
    ) -> Result<(), Self::E>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

pub trait GraphicsPass {
    type MP: MemoryPool;
    type PSetType: PipelineSet<MP = Self::MP>;
    type PAttachType: GraphicsPassAttachments<MP = Self::MP>;
    type E: Error;

    fn create_sets(&self, subpass_id: usize) -> Result<Vec<Self::PSetType>, Self::E>;
    fn create_attachments(
        &self,
        name: &str,
        allocator: &mut Self::MP,
        res: Resolution2d,
    ) -> Result<Self::PAttachType, Self::E>;
}

pub struct SubpassInfo {
    pub color_attachments: Vec<usize>,
    pub depth_attachment: Option<usize>,
    pub set_infos: Vec<Vec<PipelineSetBindingInfo>>,
    pub shaders: HashMap<ShaderType, Vec<u32>>,
    pub depends_on: Vec<usize>,
}

pub trait PipelineSet {
    type MP: MemoryPool;
    type E: Error;

    fn update_bindings(
        &mut self,
        binding_writables: Vec<PipelineSetBindingWritable<Self::MP>>,
    ) -> Result<(), Self::E>;
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineSetBindingType {
    UniformBuffer,
    StorageBuffer,
    Sampler2d,
}

#[derive(Debug, Clone)]
pub struct PipelineSetBindingInfo {
    pub _type: PipelineSetBindingType,
    pub count: usize,
    pub bindless: bool,
}

pub enum PipelineSetBindingWritable<'a, MP: MemoryPool> {
    Buffer(&'a MP, &'a [BufferId]),
    Image2d(&'a MP, &'a [Image2dId]),
}

pub trait GraphicsPassAttachments {
    type MP: MemoryPool;

    fn resolution(&self) -> Resolution2d;
    fn get_attachments(&self) -> Vec<(&Self::MP, Image2dId)>;
}

#[derive(Debug, Clone)]
pub struct BufferProps {
    gpu_local: bool,
    size: u64,
    usage: BitFlags<BufferUsage>,
}

#[derive(Debug, Clone)]
pub struct Image2dProps {
    gpu_local: bool,
    resolution: Resolution2d,
    format: ImageFormat,
    usage: BitFlags<ImageUsage>,
}

new_key_type! {
    pub struct BufferId;
    pub struct Image2dId;
}

pub trait MemoryPool {
    type E: Error;

    fn new_buffer(&mut self, props: BufferProps) -> Result<BufferId, Self::E>;
    fn get_buffer_props(&self, id: BufferId) -> Option<&BufferProps>;
    fn write_to_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), Self::E>;

    fn new_image_2d(&mut self, props: Image2dProps) -> Result<Image2dId, Self::E>;
    fn get_image_2d_props(&self, id: Image2dId) -> Option<&Image2dProps>;
}

#[bitflags]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferUsage {
    None = 1 << 0,
    Uniform = 1 << 1,
    Storage = 1 << 2,
    TransferSrc = 1 << 3,
    TransferDst = 1 << 4,
}

#[derive(Debug, Clone, Copy)]
pub struct Resolution2d {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    R32,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba8,
    D24S8,
    D32,
    D32S8,
}

impl ImageFormat {
    pub fn is_depth(&self) -> bool {
        match self {
            Self::D24S8 | Self::D32 | Self::D32S8 => true,
            _ => false,
        }
    }

    pub fn has_stencil(&self) -> bool {
        match self {
            Self::D24S8 | Self::D32S8 => true,
            _ => false,
        }
    }
}

#[bitflags]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageUsage {
    None = 1 << 0,
    CopySrc = 1 << 1,
    CopyDst = 1 << 2,
    PipelineAttachment = 1 << 3,
    Present = 1 << 4,
}

pub trait GpuFuture {}

pub trait CpuFuture {
    type E: Error;

    fn wait(&self) -> Result<(), Self::E>;
}

pub trait Swapchain {
    type MP: MemoryPool;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error;

    fn is_optimized(&self) -> bool;
    fn get_next_image(
        &mut self,
        cfut: Option<&Self::CFutType>,
        gfut: Option<&Self::GFutType>,
    ) -> Result<u32, Self::E>;
    fn refresh_resolution(&mut self) -> Result<(), Self::E>;
    fn images(&self) -> &[Image2dId];
    fn present(&self, idx: u32, wait_for: &[&Self::GFutType]) -> Result<bool, Self::E>;
}
