use crate::{
    Capped,
    buffer::Buffer,
    image::{Image, ImageView, Sampler},
};

#[derive(Debug, Clone)]
pub enum ShaderSetInfo {
    UniformBuffer(usize),
    StorageBuffer(usize),
    Sampler2D(usize),
}

pub enum ShaderSetData<B: Buffer, IV: ImageView, S: Sampler> {
    UniformBuffer(Vec<Capped<B>>),
    StorageBuffer(Vec<Capped<B>>),
    Sampler2D(Vec<(Capped<IV>, Capped<S>)>),
}

pub trait ShaderSet {
    type B: Buffer;
    type I: Image;
    type IV: ImageView<I = Self::I>;
    type S: Sampler;

    fn update_binding_data(
        &mut self,
        binding: usize,
        data: ShaderSetData<Self::B, Self::IV, Self::S>,
    );
}
