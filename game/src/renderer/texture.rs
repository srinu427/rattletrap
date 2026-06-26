use avk12::{
    pipeline::DSet,
    resource::{ImageView, Sampler},
};

pub struct Texture {
    ss: DSet,
    iv: ImageView,
    s: Sampler,
}
