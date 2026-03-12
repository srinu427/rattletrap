use enumflags2::bitflags;

use crate::HostAccess;

#[derive(Debug, Clone, thiserror::Error)]
pub enum BufferErr {
    #[error("buffer is not host writeable")]
    NotHostWriteable,
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BufferFlags {
    CopyDst,
    CopySrc,
    Vertex,
    Index,
}

pub trait Buffer: Clone {
    fn size(&self) -> usize;
    fn host_access(&self) -> HostAccess;
    fn host_write(&self, data: &[u8]) -> Result<(), BufferErr>;
}
