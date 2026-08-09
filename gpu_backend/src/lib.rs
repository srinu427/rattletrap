use std::{ops::Range, rc::Rc, sync::Arc};

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct BufferUsage: u32 {
        const COPY_SRC = 1;
        const COPY_DST = 1 << 1;
        const UNIFORM = 1 << 2;
        const STORAGE = 1 << 3;
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct ImageUsage: u32 {
        const COPY_SRC = 1;
        const COPY_DST = 1 << 1;
        const SAMPLED = 1 << 2;
        const STORAGE = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Unorm,
    Bgra8Srgb,
    Rgba10Unorm,
    Rgba16Sfloat,
    D24S8,
    D32,
}

#[derive(Debug, Clone, Copy)]
pub enum MemLocation {
    Gpu,
    CpuToGpu,
}

pub enum Captured<'a, T> {
    Obj(&'a T),
    Arc(&'a Arc<T>),
    Rc(&'a Rc<T>),
}

impl<'a, T> Captured<'a, T> {
    pub fn as_ref(&self) -> &T {
        match self {
            Captured::Obj(t) => t,
            Captured::Arc(t) => t.as_ref(),
            Captured::Rc(t) => t.as_ref(),
        }
    }
}

pub trait GpuDevice {
    type B: Buffer;
    type I: Image;
    type S: Sampler;
    type SCI: SwapchainImage<I = Self::I>;
    type BGL: BindGroupLayout;
    type BG: BindGroup<B = Self::B, I = Self::I, S = Self::S>;
    type RP: RenderPipeline;
    type CR: CommandRecorder<B = Self::B, I = Self::I>;
    fn new_buffer(&self, size: u64, usage: BufferUsage, mem_loc: MemLocation) -> Self::B;
    fn new_image(
        &self,
        format: ImageFormat,
        res: (u32, u32, u32),
        layers: u32,
        mip_levels: u32,
    ) -> Self::I;
    fn new_sampler(&self) -> Self::S;
    fn new_bind_group_layout(&self, bindings: Vec<BindingDesc>) -> Self::BGL;
    fn new_bind_group(&self, layout: &Self::BGL) -> Self::BG;
    fn new_render_pipeline<'a>(
        &self,
        vert_info: VertexStage,
        frag_info: FragmentStage,
        raster_info: RasterStage,
        depth_info: Option<DepthStage>,
        bg_layouts: Vec<Captured<Self::BGL>>,
    ) -> Self::RP;
    fn new_command_recorder(&self) -> Self::CR;
    fn run_tasks(&self, recorders: Vec<Self::CR>);
    fn get_surface_config(&self) -> SurfaceConfig;
    fn get_current_frame(&self) -> Option<Self::SCI>;
}

pub trait Buffer {
    fn size(&self) -> u64;
    fn write_data(&mut self, offset: u64, data: &[u8]);
}

pub trait Image {
    fn format(&self) -> ImageFormat;
    fn res(&self) -> (u32, u32, u32);
    fn layers(&self) -> u32;
    fn mip_levels(&self) -> u32;
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageViewType {
    V2D,
    VCube,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewInfo {
    pub type_: ImageViewType,
    pub level_range: Range<u32>,
    pub layer_range: Range<u32>,
}

pub struct ImageView<'a, I>
where
    I: Image,
{
    pub info: ImageViewInfo,
    pub image: Captured<'a, I>,
}

pub trait Sampler {}

pub struct SurfaceConfig {
    pub res: (u32, u32),
    pub format: ImageFormat,
    pub image_count: u32,
    pub vsync_preferred: bool,
}

pub trait SwapchainImage {
    type I: Image;
    fn image(&self) -> &Self::I;
}

#[derive(Debug, Clone, Copy)]
pub enum BindingType {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
}

#[derive(Debug, Clone)]
pub struct BindingDesc {
    pub type_: BindingType,
    pub len: u32,
}

pub trait BindGroupLayout {}

pub trait BindGroup {
    type B: Buffer;
    type I: Image;
    type S: Sampler;
    fn write_buffers(&mut self, binding: u32, offset: u32, buffers: Vec<Captured<Self::B>>);
    fn write_image_views<'a>(
        &mut self,
        binding: u32,
        offset: u32,
        ivs: Vec<ImageView<'a, Self::I>>,
    );
    fn write_samplers(&mut self, binding: u32, offset: u32, buffers: Vec<Captured<Self::S>>);
}

#[derive(Debug, Clone)]
pub enum ShaderSource {
    WgslStr { data: String, entrypoint: String },
    WgslFile { file: String, entrypoint: String },
    GlslStr(String),
    GlslFile(String),
    SpvWords(Vec<u32>),
    SpvFile(String),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexBindDesc {
    Vec3f,
    Vec4f,
}

#[derive(Debug)]
pub struct VertexBufferLayout {
    pub bindings: Vec<VertexBindDesc>,
    pub stride: u32,
}

#[derive(Debug)]
pub struct VertexStage {
    pub shader: ShaderSource,
    pub vertex_layout: Option<VertexBufferLayout>,
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct CullMode: u8 {
        const BACK = 1;
        const FRONT = 1 << 1;
    }
}

#[derive(Debug)]
pub struct RasterStage {
    pub front_ccw: bool,
    pub cull_mode: CullMode,
}

#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Less,
    LessOrEq,
    Greater,
    GreaterOrEq,
    Any,
}

#[derive(Debug, Clone)]
pub struct DepthStage {
    pub format: ImageFormat,
    pub compare_op: CompareOp,
    pub test: bool,
    pub clear: bool,
    pub store: bool,
}

#[derive(Debug, Clone)]
pub struct RenderColorAttachment {
    pub format: ImageFormat,
    pub clear: bool,
    pub store: bool,
}

#[derive(Debug)]
pub struct FragmentStage {
    pub shader: ShaderSource,
    pub color_attachments: Vec<RenderColorAttachment>,
}

pub trait RenderPipeline {}

pub trait CommandRecorder {
    type B: Buffer;
    type I: Image;
    fn copy_b2b(
        &mut self,
        src: Captured<Self::B>,
        src_offset: u64,
        dst: Captured<Self::B>,
        dst_offset: u64,
        len: u64,
    );
    fn copy_b2i(&mut self, src: Captured<Self::B>, src_offset: u64, dst: Captured<Self::I>);
}

pub trait RenderCommandRecorder<'a> {
    type B: Buffer;
    type I: Image;
    type BG: BindGroup;
    type CR: CommandRecorder<B = Self::B, I = Self::I>;
    type RP: RenderPipeline;
    fn from(
        cr: &'a Self::CR,
        pipeline: &'a Self::RP,
        attachments: Vec<ImageView<'a, Self::I>>,
    ) -> Self;
    fn set_vertex_buffers(&mut self, vertex_buffer: Captured<Self::B>);
    fn set_index_buffer(&mut self, buffer: Captured<Self::B>, is_16bit: bool);
    fn set_bind_group(&mut self, idx: u32, bg: &Self::BG);
    fn set_pc_data(&mut self, data: &[u8]);
    fn draw(&mut self, count: u32);
    fn draw_indexed(&mut self, idx_offset: u32, vert_offset: u32, count: u32);
}
