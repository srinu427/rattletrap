use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use thiserror::Error;

use crate::wrappers::{
    buffer::BufferError,
    image::Image,
    image_view::{ImageView, ImageViewError},
    logical_device::LogicalDevice,
};

#[derive(Debug, Error)]
pub enum TextureError {
    #[error("Image load error: {0}")]
    ImageLoadError(image::ImageError),
    #[error("Image creation error: {0}")]
    ImageError(#[from] crate::wrappers::image::ImageError),
    #[error("Image view creation error: {0}")]
    ImageCreationError(#[from] ImageViewError),
    #[error("Stage Buffer creation error: {0}")]
    StageBufferCreationError(vk::Result),
    #[error("Stage Buffer allocation error: {0}")]
    BufferError(#[from] BufferError),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Texture {
    #[get = "pub"]
    albedo: Arc<ImageView>,
}

impl Texture {
    pub fn new(albedo: Arc<ImageView>) -> Self {
        Self { albedo }
    }
}
