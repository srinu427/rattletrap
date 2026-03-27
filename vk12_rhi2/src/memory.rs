use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use log::warn;

use crate::device::DeviceDropper;

pub struct MemAlloc {
    pub allocator: Mutex<Allocator>,
    pub device_dropper: Arc<DeviceDropper>,
}
impl MemAlloc {
    pub fn new(device_dropper: &Arc<DeviceDropper>) -> Result<Self, String> {
        let allocator_create_info = AllocatorCreateDesc {
            instance: device_dropper.instance_dropper.instance.clone(),
            device: device_dropper.device.clone(),
            physical_device: device_dropper.gpu_info.handle,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };
        let allocator = Allocator::new(&allocator_create_info)
            .map_err(|e| format!("creating gpu mem allocator failed: {e}"))?;
        Ok(Self {
            allocator: Mutex::new(allocator),
            device_dropper: device_dropper.clone(),
        })
    }
}

pub struct Memory {
    pub allocator: Arc<MemAlloc>,
    pub handle: ManuallyDrop<Allocation>,
}

impl Memory {
    pub fn new(
        allocator: &Arc<MemAlloc>,
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
        let mut allocator_mut = match allocator.allocator.lock() {
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
            let mut allocator_mut = match self.allocator.allocator.lock() {
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
