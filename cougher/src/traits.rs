use core::error::Error;
use enumflags2::{BitFlags, bitflags};
use hashbrown::HashMap;

pub trait MemAllocator {}

pub trait MemAllocation {
    type AllocatorType: MemAllocator;

    fn is_gpu_local(&self) -> bool;
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

pub trait Buffer {
    type AllocatorType: MemAllocator;
    type MemType: MemAllocation<AllocatorType = Self::AllocatorType>;
    type E: Error;

    fn name(&self) -> &str;
    fn write_data(&mut self, offset: u64, data: &[u8]) -> Result<(), Self::E>;
    fn size(&self) -> u64;
    fn usage(&self) -> BitFlags<BufferUsage>;
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

pub trait Image2d {
    type AllocatorType: MemAllocator;
    type MemType: MemAllocation<AllocatorType = Self::AllocatorType>;
    type E: Error;

    fn resolution(&self) -> Resolution2d;
}

pub trait GpuFuture {}

pub trait CpuFuture {
    type E: Error;

    fn wait(&self) -> Result<(), Self::E>;
}

pub trait Swapchain {
    type Image2dType: Image2d;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error;

    fn is_optimized(&self) -> bool;
    fn get_next_image(
        &mut self,
        cfut: Option<&Self::CFutType>,
        gfut: Option<&Self::GFutType>,
    ) -> Result<u32, Self::E>;
    fn resize_resolution(&mut self) -> Result<(), Self::E>;
    fn images(&self) -> &[Self::Image2dType];
    fn present(&self, idx: u32, wait_for: &[&Self::GFutType]) -> Result<bool, Self::E>;
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

pub enum PipelineSetBindingWritable<'a, B: Buffer, I: Image2d> {
    Buffer(&'a [B]),
    Image2d(&'a [I]),
}

pub trait GraphicsPassAttachments {
    type I2dType: Image2d;

    fn resolution(&self) -> Resolution2d;
    fn get_attachments(&self) -> Vec<&Self::I2dType>;
}

pub trait PipelineSet {
    type BType: Buffer;
    type I2dType: Image2d;
    type E: Error;

    fn update_bindings(
        &mut self,
        binding_writables: Vec<PipelineSetBindingWritable<Self::BType, Self::I2dType>>,
    ) -> Result<(), Self::E>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

pub trait GraphicsPass {
    type AllocatorType: MemAllocator;
    type MemType: MemAllocation<AllocatorType = Self::AllocatorType>;
    type BType: Buffer<MemType = Self::MemType>;
    type I2dType: Image2d<MemType = Self::MemType>;
    type PSetType: PipelineSet<BType = Self::BType, I2dType = Self::I2dType>;
    type PAttachType: GraphicsPassAttachments<I2dType = Self::I2dType>;
    type E: Error;

    fn create_sets(&self, subpass_id: usize) -> Result<Vec<Self::PSetType>, Self::E>;
    fn create_attachments(
        &self,
        name: &str,
        allocator: &mut Self::AllocatorType,
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum QueueType {
    Graphics,
}

pub enum GpuCommand<'a, B: Buffer, I2d: Image2d> {
    Image2dUsageHint { image: &'a I2d, usage: ImageUsage },
    CopyBufferToImage2d { src: &'a B, dst: &'a I2d },
    BlitImage2d { src: &'a I2d, dst: &'a I2d },
}

pub struct GpuExecutorInput<'a, S: GpuFuture, F: CpuFuture> {
    pub cmd_list: &'a str,
    pub wait_for: Vec<&'a S>,
    pub emit_gfuts: Vec<&'a S>,
    pub emit_cfut: Option<&'a F>,
}

pub trait GpuExecutor {
    type BType: Buffer;
    type I2dType: Image2d;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error;

    fn type_(&self) -> QueueType;
    fn new_command_list(&mut self, name: &str) -> Result<(), Self::E>;
    fn update_command_list(
        &mut self,
        name: &str,
        commands: Vec<GpuCommand<Self::BType, Self::I2dType>>,
    ) -> Result<(), Self::E>;
    fn run_command_lists(
        &self,
        lists: &[&str],
        wait_for: Vec<&Self::GFutType>,
        emit_gfuts: Vec<&Self::GFutType>,
        emit_cfut: Option<&Self::CFutType>,
    ) -> Result<(), Self::E>;
}

pub trait GpuContext {
    type AllocatorType: MemAllocator;
    type AllocationType: MemAllocation<AllocatorType = Self::AllocatorType>;
    type BufferType: Buffer<MemType = Self::AllocationType, AllocatorType = Self::AllocatorType>;
    type Image2dType: Image2d<MemType = Self::AllocationType, AllocatorType = Self::AllocatorType>;
    type SwapchainType: Swapchain<
            Image2dType = Self::Image2dType,
            GFutType = Self::GFutType,
            CFutType = Self::CFutType,
        >;
    type PSetType: PipelineSet<BType = Self::BufferType, I2dType = Self::Image2dType>;
    type PAttachType: GraphicsPassAttachments<I2dType = Self::Image2dType>;
    type QType: GpuExecutor<
            BType = Self::BufferType,
            I2dType = Self::Image2dType,
            GFutType = Self::GFutType,
            CFutType = Self::CFutType,
        >;
    type GPassType: GraphicsPass<
            AllocatorType = Self::AllocatorType,
            PSetType = Self::PSetType,
            PAttachType = Self::PAttachType,
        >;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error
        + From<image::ImageError>
        + From<<Self::GPassType as GraphicsPass>::E>
        + From<<Self::SwapchainType as Swapchain>::E>
        + From<<Self::CFutType as CpuFuture>::E>
        + From<<Self::QType as GpuExecutor>::E>
        + From<<Self::BufferType as Buffer>::E>
        + From<<Self::Image2dType as Image2d>::E>;

    fn new_buffer(
        &self,
        allocator: &mut Self::AllocatorType,
        gpu_local: bool,
        size: u64,
        name: &str,
        usage: BitFlags<BufferUsage>,
    ) -> Result<Self::BufferType, Self::E>;
    fn new_image_2d(
        &self,
        allocator: &mut Self::AllocatorType,
        gpu_local: bool,
        name: &str,
        resolution: Resolution2d,
        format: ImageFormat,
        usage: BitFlags<ImageUsage>,
    ) -> Result<Self::Image2dType, Self::E>;
    fn new_allocator(&self) -> Result<Self::AllocatorType, Self::E>;
    fn new_swapchain(&self, usages: BitFlags<ImageUsage>) -> Result<Self::SwapchainType, Self::E>;
    fn new_gpu_future(&self) -> Result<Self::GFutType, Self::E>;
    fn new_cpu_future(&self, signaled: bool) -> Result<Self::CFutType, Self::E>;
    fn new_graphics_pass(
        &self,
        attachments: Vec<ImageFormat>,
        subpass_infos: Vec<SubpassInfo>,
        max_sets: usize,
    ) -> Result<Self::GPassType, Self::E>;
    fn get_queue(&mut self) -> Result<Self::QType, Self::E>;
}

pub trait GpuInfo {
    fn name(&self) -> String;
    fn vram(&self) -> u64;
    fn is_dedicated(&self) -> bool;
}

pub trait ApiLoader {
    type GpuInfoType: GpuInfo;
    type ContextType: GpuContext;
    type E: Error;

    fn list_supported_gpus(&self) -> Vec<Self::GpuInfoType>;
    fn new_gpu_context(self, gpu: Self::GpuInfoType) -> Result<Self::ContextType, Self::E>;
}
