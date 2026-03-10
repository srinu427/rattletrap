use enumflags2::{BitFlags, bitflags};

pub use enumflags2;

pub enum BufferErr {
    NotHostWriteable,
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BufferFlags {
    HostAccess,
    CopyDst,
    CopySrc,
    Vertex,
    Index,
}

pub trait Buffer {
    fn size(&self) -> usize;
    fn host_writeable(&self) -> usize;
    fn host_write(&self) -> Result<(), BufferErr>;
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum ImageFlags {
    HostAccess,
    CopyDst,
    CopySrc,
    RenderAttach,
    Sampled,
}

pub enum ImageDimension {
    E2d,
    E3d,
}

pub trait Image {
    fn res(&self) -> (u32, u32, u32);
}

pub trait Device {
    type BType: Buffer;
    fn new_buffer(&self, size: usize, flags: BitFlags<BufferFlags>) -> Self::BType;
    fn new_image(&self, dim: ImageDimension, res: (u32, u32, u32));
}
