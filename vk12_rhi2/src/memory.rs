use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use log::warn;

pub struct Memory {
    pub allocator: Arc<Mutex<Allocator>>,
    pub handle: ManuallyDrop<Allocation>,
}

impl Memory {
    pub fn new(
        allocator: &Arc<Mutex<Allocator>>,
        reqs: vk::MemoryRequirements,
        name: &str,
        host_access: rhi2::HostAccess,
    ) -> Result<Self, String> {
        let location = match host_access {
            rhi2::HostAccess::None => MemoryLocation::GpuOnly,
            rhi2::HostAccess::Read => MemoryLocation::GpuToCpu,
            rhi2::HostAccess::Write => MemoryLocation::CpuToGpu,
        };
        let create_info = AllocationCreateDesc {
            name,
            requirements: reqs,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let mut allocator_mut = match allocator.lock() {
            Ok(obj) => obj,
            Err(e) => e.into_inner(),
        };
        let allocation = allocator_mut
            .allocate(&create_info)
            .map_err(|e| format!("gpu mem allocation failed: {e}"))?;
        let handle = ManuallyDrop::new(allocation);
        Ok(Self {
            allocator: allocator.clone(),
            handle,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        unsafe {
            let mem = ManuallyDrop::take(&mut self.handle);
            let mut allocator_mut = match self.allocator.lock() {
                Ok(obj) => obj,
                Err(e) => e.into_inner(),
            };
            allocator_mut
                .free(mem)
                .inspect_err(|e| warn!("allocation freeing failed: {e}"))
                .ok();
        }
    }
}
