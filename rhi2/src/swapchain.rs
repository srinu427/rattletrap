use crate::{
    command::CommandRecorder,
    image::{Format, Image, ImageView},
    sync::TaskFuture,
};

#[derive(Debug, thiserror::Error)]
pub enum SwapchainErr {
    #[error("acquiring swapchain image failed: {0}")]
    NextImageIdxErr(String),
    #[error("refreshing swapchain res failed: {0}")]
    ResRefreshErr(String),
    #[error("present swapchain image failed: {0}")]
    PresentImageErr(String),
}

pub trait SwapchainImage {
    type I: Image;
    type IV: ImageView<I = Self::I>;

    fn view(&self) -> &Self::IV;
    fn present(&mut self) -> Result<(), SwapchainErr>;
}

pub enum SCImageRes<SI: SwapchainImage> {
    Success(SI),
    Unavailable,
    Outdated,
    Error(String),
}

pub trait Swapchain {
    type I: Image;
    type IV: ImageView<I = Self::I>;
    type TF: TaskFuture;
    type CR: CommandRecorder<I = Self::I, IV = Self::IV, TF = Self::TF>;
    type SI: SwapchainImage<I = Self::I, IV = Self::IV>;

    fn res(&self) -> (u32, u32);
    fn fmt(&self) -> Format;
    fn img_count(&self) -> usize;
    fn refresh_res(&mut self) -> Result<(), SwapchainErr>;
    fn next_image(&mut self) -> SCImageRes<Self::SI>;
}
