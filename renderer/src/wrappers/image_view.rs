use std::sync::Arc;

use ash::vk;

use crate::wrappers::image::Image;

#[derive(Debug, thiserror::Error)]
pub enum ImageViewError {
    #[error("Image view creation error: {0}")]
    CreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct ImageView {
    #[get_copy = "pub"]
    image_view: vk::ImageView,
    #[get_copy = "pub"]
    type_: vk::ImageViewType,
    #[get = "pub"]
    image: Arc<Image>,
}

impl ImageView {
    pub fn new(
        image: Arc<Image>,
        type_: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Result<Self, ImageViewError> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image.image())
            .view_type(type_)
            .format(image.format())
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .subresource_range(subresource_range);

        let image_view = unsafe {
            image
                .device()
                .device()
                .create_image_view(&create_info, None)
                .map_err(ImageViewError::CreateError)?
        };
        Ok(Self {
            image_view,
            type_,
            image,
        })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.image
                .device()
                .device()
                .destroy_image_view(self.image_view, None);
        }
    }
}
