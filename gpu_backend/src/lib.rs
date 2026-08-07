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
    fn new_buffer(&self, size: u64, usage: BufferUsage, mem_loc: MemLocation) -> Self::B;
    fn new_image(
        &self,
        format: ImageFormat,
        res: (u32, u32, u32),
        layers: u32,
        mip_levels: u32,
    ) -> Self::I;
    fn new_sampler(&self) -> Self::S;
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
    pub idx: u32,
    pub type_: BindingType,
    pub len: u32,
}

pub trait BindGroupLayout {}

pub trait BindGroup {}

pub trait RenderPipeline {}

pub trait CommandRecorder {}

pub trait RenderCommandRecorder {}
