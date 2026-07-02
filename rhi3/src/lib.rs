use std::ops::Range;

use bitflags::bitflags;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub dvram: u64,
    pub is_dedicated: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryLocation {
    Gpu,
    CpuToGpu,
    GpuToCpu,
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsageFlags: u32 {
        const UNIFORM = 1;
        const STORAGE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Rgba8,
    Bgra8,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba10,
    Bgra10,
    Rgba16,
    Bgra16,
    D16,
    D16S8,
    D24,
    D24S8,
    D32,
}

impl Format {
    pub fn is_depth(&self) -> bool {
        match self {
            Self::D16 | Self::D16S8 | Self::D24 | Self::D24S8 | Self::D32 => true,
            _ => false,
        }
    }

    pub fn has_stencil(&self) -> bool {
        match self {
            Self::D16S8 | Self::D24S8 => true,
            _ => false,
        }
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageUsageFlags: u32 {
        const SAMPLED = 1;
        const STORAGE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const ATTACHMENT = 1 << 4;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageViewType {
    E2d,
    ECube,
}

pub trait Initializer {
    type C: Client;
    fn get_gpus(&self) -> Vec<GpuInfo>;
    fn make_client(&self, gpu_id: usize) -> anyhow::Result<Self::C>;
}

pub trait Client {
    type B: Buffer;
    type IV: ImageView<I = Self::I>;
    type I: Image<IV = Self::IV>;
    type S: Sampler;
    fn gpu_info(&self) -> GpuInfo;
    fn new_buffer(
        &self,
        size: u64,
        mem_location: MemoryLocation,
        usage_flags: BufferUsageFlags,
    ) -> anyhow::Result<Self::B>;
    fn new_image(
        &self,
        format: Format,
        res: (u32, u32, u32),
        layers: u32,
        levels: u32,
        mem_location: MemoryLocation,
        usage_flags: ImageUsageFlags,
    ) -> anyhow::Result<Self::I>;
    fn sampler(&self) -> anyhow::Result<Self::S>;
}

pub trait Buffer {
    fn size(&self) -> u64;
    fn cpu_write(&mut self, data: &[u8]) -> anyhow::Result<()>;
}

pub trait Image {
    type IV: ImageView<I = Self>;
    fn format(&self) -> Format;
    fn res(&self) -> (u32, u32, u32);
    fn layers(&self) -> u32;
    fn levels(&self) -> u32;
    fn view(
        &self,
        type_: ImageViewType,
        layer_range: Range<u32>,
        level_range: Range<u32>,
    ) -> anyhow::Result<Self::IV>;
}

pub trait ImageView {
    type I: Image<IV = Self>;
    fn type_(&self) -> ImageViewType;
    fn image(&self) -> &Self::I;
}

pub trait Sampler {}

pub enum TaskFutureResult {
    Success,
    Timeout,
    Error(String),
}

pub trait TaskFuture {
    fn wait(&self, timeout: u64) -> TaskFutureResult;
}

pub trait GraphicsTask<'a> {
    type B: Buffer;
    type IV: ImageView<I = Self::I>;
    type I: Image<IV = Self::IV>;
    type S: Sampler;
}

pub trait Task {
    type B: Buffer;
    type IV: ImageView<I = Self::I>;
    type I: Image<IV = Self::IV>;
    type S: Sampler;
    type GT<'a>: GraphicsTask<'a, B = Self::B, I = Self::I, IV = Self::IV, S = Self::S>
    where
        Self: 'a;

    fn copy_b2b(
        &mut self,
        src: &Self::B,
        src_offset: u64,
        dst: &Self::B,
        dst_offset: u64,
        len: u64,
    );
    fn copy_b2i(&mut self, src: &Self::B, src_offset: u64, dst: &Self::I);
    fn graphics(&mut self) -> Self::GT<'_>;
    fn run(self) -> TaskFutureResult;
}
