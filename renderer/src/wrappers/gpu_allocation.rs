use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{Allocation, AllocationScheme, Allocator},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuAllocationError {
    #[error("GPU Allocator error: {0}")]
    AllocatorError(#[from] AllocationError),
    #[error("Mutex lock error")]
    LockError,
}

#[derive(getset::Getters, getset::MutGetters)]
pub struct GpuAllocation {
    #[get = "pub"]
    #[get_mut = "pub"]
    allocation: ManuallyDrop<Allocation>,
    allocator: Arc<Mutex<Allocator>>,
}

impl GpuAllocation {
    pub fn new(
        allocator: Arc<Mutex<Allocator>>,
        name: &str,
        requirements: vk::MemoryRequirements,
        linear: bool,
        location: MemoryLocation,
    ) -> Result<Self, GpuAllocationError> {
        let mut allocator_locked = allocator
            .lock()
            .map_err(|_| GpuAllocationError::LockError)?;
        let allocation =
            allocator_locked.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements,
                location,
                linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        Ok(Self {
            allocation: ManuallyDrop::new(allocation),
            allocator: allocator.clone(),
        })
    }
}

impl Drop for GpuAllocation {
    fn drop(&mut self) {
        let Ok(mut allocator) = self.allocator.lock() else {
            return;
        };
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        allocator.free(allocation).ok();
    }
}
