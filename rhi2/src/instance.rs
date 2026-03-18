use crate::{
    buffer::Buffer,
    command::{CommandRecorder, GraphicsCommandRecorder},
    device::Device,
    graphics_pipeline::GraphicsPipeline,
    image::{Image, ImageView},
    shader::ShaderSet,
    swapchain::Swapchain,
    sync::TaskFuture,
};

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub dvram: u64,
    pub is_dedicated: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InstanceErr {
    #[error("device creation failed: {0}")]
    DeviceCreateFailed(String),
}

pub trait Instance {
    type SC: Swapchain;
    type B: Buffer;
    type I: Image;
    type IV: ImageView<I = Self::I>;
    type SS: ShaderSet<B = Self::B, I = Self::I, IV = Self::IV>;
    type GP: GraphicsPipeline<B = Self::B, I = Self::I, IV = Self::IV, SS = Self::SS>;
    type GCR: GraphicsCommandRecorder<B = Self::B, I = Self::I, IV = Self::IV, SS = Self::SS>;
    type CR: CommandRecorder<
            B = Self::B,
            I = Self::I,
            IV = Self::IV,
            SS = Self::SS,
            GP = Self::GP,
            GCR = Self::GCR,
            TF = Self::TF,
        >;
    type TF: TaskFuture;
    type DType: Device<
            B = Self::B,
            SC = Self::SC,
            I = Self::I,
            IV = Self::IV,
            SS = Self::SS,
            GP = Self::GP,
            CR = Self::CR,
            TF = Self::TF,
        >;

    fn get_gpus(&self) -> &Vec<GpuInfo>;
    fn init_device(self, gpu_id: usize) -> Result<Self::DType, InstanceErr>;
}
