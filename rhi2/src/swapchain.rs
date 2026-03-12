use crate::image::{Image, ImageView};

pub trait Swapchain {
    type IType: Image;
    type IVType: ImageView<IType = Self::IType>;
    fn img_count(&self) -> usize;
    fn res(&self) -> (u32, u32);
    fn get_image(&self, num: usize) -> &Self::IType;
    fn get_image_view(&self, num: usize) -> &Self::IVType;
    fn refresh_res(&mut self);
}
