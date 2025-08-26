use std::sync::Arc;

use thiserror::Error;
use winit::window::Window;

use crate::wrappers::{
    instance::{Instance, InstanceError},
    logical_device::{LogicalDevice, LogicalDeviceInitError},
};

#[derive(Debug, Error)]
pub enum ContextInitError {
    #[error("Instance creation error: {0}")]
    InstanceError(#[from] InstanceError),
    #[error("Logical device creation error: {0}")]
    LogicalDeviceError(#[from] LogicalDeviceInitError),
}

struct Context {
    pub(crate) instance: Arc<Instance>,
    pub(crate) window: Arc<Window>,
}

impl Context {
    pub fn new(window: Arc<Window>) -> Result<Self, ContextInitError> {
        let instance = Arc::new(Instance::new(window.clone())?);
        let logical_device = LogicalDevice::new(instance.clone(), window.clone())?;
        todo!()
    }
}
