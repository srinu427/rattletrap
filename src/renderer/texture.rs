use std::sync::Arc;

use rhi2::device::Device;

pub struct Texture<D: Device> {
    ss: D::SS,
    iv: D::IV,
    s: Arc<D::S>,
}
