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

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Context {
    #[get = "pub"]
    logical_device: Arc<LogicalDevice>,
    #[get = "pub"]
    instance: Arc<Instance>,
    #[get = "pub"]
    window: Arc<Window>,
}

impl Context {
    pub fn new(window: Arc<Window>) -> Result<Self, ContextInitError> {
        let instance = Arc::new(Instance::new(window.clone())?);
        let logical_device = Arc::new(LogicalDevice::new(instance.clone())?);

        Ok(Self {
            logical_device,
            instance,
            window,
        })
    }
}
