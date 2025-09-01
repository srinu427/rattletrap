use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use thiserror::Error;

use crate::wrappers::{buffer::BufferError, image::Image, image_view::{ImageView, ImageViewError}, logical_device::LogicalDevice};

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
    pub fn from_path(
        path: &Path,
        device: Arc<LogicalDevice>,
        allocator: Arc<Mutex<Allocator>>,
    ) -> Result<Self, TextureError> {
        let img = image::open(path).map_err(TextureError::ImageLoadError)?;
        let extent = vk::Extent2D {
            width: img.width(),
            height: img.height(),
        };

        let mut image = Image::new_2d(
            device.clone(),
            vk::Format::R8G8B8A8_SRGB,
            extent,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;

        image.allocate_memory(allocator.clone(), true)?;

        let image = Arc::new(image);

        let image_view = ImageView::new(
            image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .map(Arc::new)?;

        Ok(Self { albedo: image_view })
    }
}
