use std::sync::Arc;

use crate::{
    buffer::Buffer,
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

pub trait Swapchain {
    type B: Buffer;
    type I: Image;
    type IV: ImageView<I = Self::I>;
    type TF: TaskFuture;
    type CR: CommandRecorder<B = Self::B, I = Self::I, IV = Self::IV, TF = Self::TF>;

    fn res(&self) -> (u32, u32);
    fn fmt(&self) -> Format;
    fn views(&self) -> &[Arc<Self::IV>];
    fn refresh_res(&mut self) -> Result<(), SwapchainErr>;
    fn next_image_idx(&mut self) -> Result<Option<(usize, Self::TF)>, SwapchainErr>;
    fn present(&mut self, idx: usize, deps: Vec<Self::TF>) -> Result<(), SwapchainErr>;
}
