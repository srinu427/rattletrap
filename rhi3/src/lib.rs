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
        const VERTEX = 1 << 4;
        const INDEX = 1 << 5;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy)]
    pub struct RasterSide: u8 {
        const COUNTER_CLOCKWISE = 0x1;
        const CLOCKWISE = 0x10;
        const BOTH = 0x11;
    }
}

#[derive(Debug)]
pub enum ShaderCode {
    GlslFile(String),
    GlslBytes(Vec<u8>),
}

#[derive(Debug, Clone, Copy)]
pub enum RasterMode {
    Fill,
    Line(u32),
}

#[derive(Debug, Clone)]
pub struct RasterizerConfig {
    pub mode: RasterMode,
    pub draw_side: RasterSide,
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
    type GR<V: VertexBufferCompat>: GraphicsRunner<V>;
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
    fn new_sampler(&self) -> anyhow::Result<Self::S>;
    fn new_graphics_runner<V: VertexBufferCompat>(
        &self,
        attachments: Vec<ShaderAttachmentInfo>,
        bind_group_layouts: Vec<Vec<BindingInfo>>,
        vertex_shader: ShaderCode,
        fragment_shader: ShaderCode,
        raster_config: RasterizerConfig,
    ) -> anyhow::Result<Self::GR<V>>;
}

pub trait Buffer {
    fn size(&self) -> u64;
    fn cpu_write(&mut self, data: &[u8]) -> anyhow::Result<()>;
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

#[derive(Debug, Clone, Copy)]
pub enum ImageViewType {
    E2d,
    ECube,
}

pub trait ImageView {
    type I: Image<IV = Self>;
    fn type_(&self) -> ImageViewType;
    fn layer_range(&self) -> Range<u32>;
    fn level_range(&self) -> Range<u32>;
    fn image(&self) -> &Self::I;
}

pub trait Sheet<'a> {
    type IV: ImageView;
    fn get_res(&self) -> (u32, u32);
    fn get_image_view(&self) -> &Self::IV;
    fn present(&self) -> anyhow::Result<()>;
}

pub enum GetSheetRes<SH> {
    Ok(SH),
    ResizeNeeded,
    Error(String),
}

pub trait Canvas {
    type IV: ImageView;
    type SH<'a>: Sheet<'a, IV = Self::IV>
    where
        Self: 'a;
    fn refresh_size(&mut self, res: (u32, u32)) -> anyhow::Result<()>;
    fn image_count(&self) -> usize;
    fn get_current_sheet(&mut self) -> GetSheetRes<Self::SH<'_>>;
}

pub trait Sampler {}

#[derive(Debug, Clone, Copy)]
pub enum BindingType {
    UniformBuffer,
    StorageBuffer,
    Texture,
    Sampler,
    TextureAndSampler,
}

#[derive(Debug, Clone, Copy)]
pub struct BindingInfo {
    pub type_: BindingType,
    pub count: usize,
}

pub trait BindGroup {
    type B: Buffer;
    type IV: ImageView;
    type S: Sampler;
    type BGU<'a>: BindGroupUpdater<'a, B = Self::B, IV = Self::IV, S = Self::S>;
    fn binding_infos(&self) -> Vec<BindingInfo>;
}

pub trait BindGroupUpdater<'a> {
    type B: Buffer;
    type IV: ImageView;
    type S: Sampler;

    fn update_binding_data_buffers(&mut self, bind_idx: usize, offset: usize, data: Vec<&Self::B>);
    fn update_binding_data_image_views(
        &mut self,
        bind_idx: usize,
        offset: usize,
        data: Vec<&Self::IV>,
    );
    fn update_binding_data_image_views_and_samplers(
        &mut self,
        bind_idx: usize,
        offset: usize,
        data: Vec<(&Self::IV, &Self::S)>,
    );
    fn flush(&mut self);
}

pub trait VertexBufferCompat: Sized {
    fn location_offsets() -> Vec<usize>;
}

#[derive(Debug, Clone, Copy)]
pub struct ShaderAttachmentInfo {
    pub format: Format,
    pub store: bool,
    pub load: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ClearValue {
    Depth(f32),
    Color([f32; 4]),
}

pub trait GraphicsRunner<V: VertexBufferCompat> {
    type B: Buffer;
    type IV: ImageView;
    type S: Sampler;
    type BG: BindGroup<B = Self::B, IV = Self::IV, S = Self::S>;
    fn new_bind_group(&self, bind_idx: usize) -> anyhow::Result<Self::BG>;
}

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
    type BG: BindGroup;

    fn set_vertex_buffer(&mut self, buffer: &Self::B);
    fn set_index_buffer(&mut self, buffer: &Self::B, is_16_bit: bool);
    fn set_bind_groups(&mut self, offset: usize, bind_groups: Vec<&Self::BG>);
    fn set_pc_data(&mut self, data: &[u8]);
    fn draw(&mut self, vert_offset: u32, count: u32);
    fn draw_indexed(&mut self, vert_offset: u32, indx_offset: u32, count: u32);
}

pub trait Task {
    type B: Buffer;
    type IV: ImageView<I = Self::I>;
    type I: Image<IV = Self::IV>;
    type S: Sampler;
    type GR<V: VertexBufferCompat>: GraphicsRunner<V, B = Self::B, IV = Self::IV, S = Self::S>;
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
    fn graphics<V: VertexBufferCompat, P: Sized + Default>(
        &mut self,
        runner: &Self::GR<V>,
        attachments: Vec<&Self::IV>,
        clear_values: Vec<ClearValue>,
    ) -> Self::GT<'_>;
    fn run(self) -> TaskFutureResult;
}
