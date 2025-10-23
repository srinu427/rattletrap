use core::error::Error;
use enumflags2::{BitFlags, bitflags};

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
    fn format(&self) -> ImageFormat;
    fn usage(&self) -> BitFlags<ImageUsage>;
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
pub enum QueueType {
    Graphics,
}

pub trait CommandBuffer {
    type BufferType: Buffer;
    type Image2dType: Image2d;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error;

    fn queue_type(&self) -> QueueType;

    fn add_image_2d_optimize_cmd(&mut self, image: &Self::Image2dType, usage: ImageUsage);
    fn copy_buffer_to_image_2d_cmd(&mut self, buffer: &Self::BufferType, image: &Self::Image2dType);
    fn add_blit_image_2d_cmd(&mut self, src: &Self::Image2dType, dst: &Self::Image2dType);

    fn build(&mut self) -> Result<(), Self::E>;
    fn reset(&mut self) -> Result<(), Self::E>;

    fn add_wait_for_gpu_future(&mut self, fut: &Self::GFutType);
    fn emit_gpu_future_on_finish(&mut self, fut: &Self::GFutType);
    fn emit_cpu_future_on_finish(&mut self, fut: &Self::CFutType);

    fn submit(&self) -> Result<(), Self::E>;
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
    type CommandBufferType: CommandBuffer<
            BufferType = Self::BufferType,
            Image2dType = Self::Image2dType,
            GFutType = Self::GFutType,
            CFutType = Self::CFutType,
        >;
    type GFutType: GpuFuture;
    type CFutType: CpuFuture;
    type E: Error
        + From<image::ImageError>
        + From<<Self::SwapchainType as Swapchain>::E>
        + From<<Self::CFutType as CpuFuture>::E>
        + From<<Self::CommandBufferType as CommandBuffer>::E>
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
    fn new_command_buffer(&self, queue_type: QueueType)
    -> Result<Self::CommandBufferType, Self::E>;
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
