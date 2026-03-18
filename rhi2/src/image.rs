use enumflags2::bitflags;

use crate::Capped;

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ImageFlags {
    CopyDst,
    CopySrc,
    RenderAttach,
    Sampled,
    Storage,
}

#[derive(Debug, Copy, Clone)]
pub enum Format {
    Rgba8,
    Bgra8,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba10,
    Bgra10,
    Rgba16,
    Rgba16Float,
    D24S8,
    D32Float,
}

impl Format {
    pub fn is_depth(&self) -> bool {
        match self {
            Self::D24S8 | Self::D32Float => true,
            _ => false,
        }
    }

    pub fn has_stencil(&self) -> bool {
        match self {
            Self::D24S8 => true,
            _ => false,
        }
    }

    pub fn rem_srgb(&self) -> Self {
        match self {
            Self::Rgba8Srgb => Self::Rgba8,
            Self::Bgra8Srgb => Self::Bgra8,
            _ => self.clone(),
        }
    }
}

pub trait Image {
    fn format(&self) -> Format;
    fn res(&self) -> (u32, u32, u32);
    fn layers(&self) -> u32;
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ImageViewErr {
    #[error("image view type requested is incompatible with the image")]
    IncompatibleViewType,
    #[error("image view create error: {0}")]
    CreateError(String),
}

#[derive(Debug, Copy, Clone)]
pub enum ViewType {
    E2d,
    ECube,
}

pub trait ImageView: Sized {
    type I: Image;

    fn new(image: Capped<Self::I>, view_type: ViewType) -> Result<Self, ImageViewErr>;
    fn image(&self) -> &Capped<Self::I>;
    fn view_type(&self) -> ViewType;
}
