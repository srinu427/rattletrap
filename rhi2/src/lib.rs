use std::{rc::Rc, sync::Arc};

pub use enumflags2;

pub mod buffer;
pub mod command;
pub mod device;
pub mod graphics_pipeline;
pub mod image;
pub mod instance;
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
