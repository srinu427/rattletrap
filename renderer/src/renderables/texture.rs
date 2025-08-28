use std::sync::Arc;

use crate::wrappers::image_view::ImageView;

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Texture {
    #[get = "pub"]
    albedo: Arc<ImageView>,
}
