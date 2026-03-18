use crate::{
    buffer::Buffer,
    image::{Format, Image, ImageView},
    shader::ShaderSet,
};

#[derive(Debug, thiserror::Error)]
pub enum GraphicsPipelineErr {
    #[error("set create failed: {0}")]
    SetCreateFailed(String),
    #[error("attach create failed: {0}")]
    AttachCreateFailed(String),
}

#[derive(Debug, Clone, Copy)]
pub enum VertexAttribute {
    Vec3,
    Vec4,
}

impl VertexAttribute {
    pub fn size(&self) -> u32 {
        match self {
            VertexAttribute::Vec3 => 3 * 4,
            VertexAttribute::Vec4 => 4 * 4,
        }
    }
}

pub struct VertexStageInfo<'a> {
    pub entrypoint: &'a str,
    pub attribs: Vec<VertexAttribute>,
    pub stride: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct AttachInfo {
    pub format: Format,
    pub clear: bool,
    pub store: bool,
}

pub struct FragmentStageInfo<'a> {
    pub entrypoint: &'a str,
    pub outputs: Vec<AttachInfo>,
    pub depth: Option<AttachInfo>,
}

pub trait GraphicsPipeline {
    type B: Buffer;
    type I: Image;
    type IV: ImageView<I = Self::I>;
    type SS: ShaderSet<B = Self::B, I = Self::I, IV = Self::IV>;

    fn set_count(&self) -> usize;
    fn pc_size(&self) -> usize;
    fn new_set(&mut self, set_id: usize) -> Result<Self::SS, GraphicsPipelineErr>;
}
