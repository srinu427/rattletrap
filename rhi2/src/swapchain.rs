use crate::{
    command::CommandRecorder,
    image::{Format, Image, ImageView},
    sync::{CpuFuture, GpuFuture},
};

#[derive(Debug, thiserror::Error)]
pub enum SwapchainErr {
    #[error("acquiring swapchain image failed: {0}")]
    AcquireImageErr(String),
    #[error("swapchain refresh needed")]
    RefreshSwapchainNeeded,
}

pub trait Swapchain {
    type I: Image;
    type IV: ImageView;
    type CR: CommandRecorder;
    type CF: CpuFuture;
    type GF: GpuFuture;

    fn res(&self) -> (u32, u32);
    fn fmt(&self) -> Format;
    fn img_count(&self) -> usize;
    fn refresh_res(&mut self);
    fn acquire_image_view(&self) -> Result<(&Self::IV, Self::GF), SwapchainErr>;
    fn present(&self, wait_for: Vec<Self::CR>) -> Result<Self::CF, SwapchainErr>;
}
