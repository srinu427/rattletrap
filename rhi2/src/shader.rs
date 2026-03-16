use crate::{
    Capped,
    buffer::Buffer,
    image::{Image, ImageView},
};

#[derive(Debug, Clone)]
pub enum ShaderSetInfo {
    UniformBuffer(usize),
    StorageBuffer(usize),
    Sampler2D(usize),
}

pub enum ShaderSetData<B: Buffer, IV: ImageView> {
    UniformBuffer(Vec<Capped<B>>),
    StorageBuffer(Vec<Capped<B>>),
    Sampler2D(Vec<Capped<IV>>),
}

pub trait ShaderSet {
    type BType: Buffer;
    type IType: Image;
    type IVType: ImageView<IType = Self::IType>;

    fn update_binding(&mut self, binding: usize, data: ShaderSetData<Self::BType, Self::IVType>);
}
