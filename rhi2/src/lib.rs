use std::{rc::Rc, sync::Arc};

pub use enumflags2;

use crate::device::Device;

pub mod buffer;
pub mod command;
pub mod device;
pub mod graphics_pipeline;
pub mod image;
pub mod shader;
pub mod swapchain;
pub mod sync;

pub enum Capped<T> {
    Obj(T),
    Arc(Arc<T>),
    Rc(Rc<T>),
}

impl<T> Capped<T> {
    pub fn from_obj(obj: T) -> Self {
        Self::Obj(obj)
    }

    pub fn from_arc(obj: Arc<T>) -> Self {
        Self::Arc(obj)
    }

    pub fn from_rc(obj: Rc<T>) -> Self {
        Self::Rc(obj)
    }

    pub fn as_ref(&self) -> &T {
        match self {
            Capped::Obj(t) => t,
            Capped::Arc(t) => t.as_ref(),
            Capped::Rc(t) => t.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HostAccess {
    None,
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub dvram: u64,
    pub is_dedicated: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InstanceErr {
    #[error("device creation failed: {0}")]
    DeviceCreateFailed(String),
}

pub trait Instance {
    type DType: Device;
    fn get_gpus(&self) -> &Vec<GpuInfo>;
    fn init_device(self, gpu_id: usize) -> Result<Self::DType, InstanceErr>;
}
