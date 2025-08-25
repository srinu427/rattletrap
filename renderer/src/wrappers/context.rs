use std::sync::Arc;

use thiserror::Error;
use winit::window::Window;

use crate::wrappers::instance::{Instance, InstanceError};

#[derive(Debug, Error)]
pub enum ContextInitError {
    #[error("Instance creation error: {0}")]
    InstanceError(#[from] InstanceError),
}

struct Context {
    instance: Arc<Instance>,
    window: Arc<Window>,
}

impl Context {
    pub fn new(window: Arc<Window>) -> Result<Self, ContextInitError> {
        let instance = Arc::new(Instance::new()?);
        
        todo!()
    }
}
