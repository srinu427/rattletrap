use std::sync::Mutex;

use gpu_allocator::{vulkan::{Allocation, Allocator, AllocatorCreateDesc}, AllocationError};
use thiserror::Error;

use crate::wrappers::logical_device::LogicalDevice;

pub struct GpuAllocation {
    pub(crate) allocation: 
}