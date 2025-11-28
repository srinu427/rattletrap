use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk::{self, Handle};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

use crate::vk_wrap::device::{AllocError, Device};

#[derive(Debug, thiserror::Error)]
pub enum BufferError {
    #[error("Error creating Vulkan Buffer: {0}")]
    CreateError(vk::Result),
    #[error("Error allocation memory for  Vulkan Image: {0}")]
    AllocationError(AllocError),
    #[error("Error binding memeory to Buffer: {0}")]
    MemoryBindError(vk::Result),
    #[error("Buffer is not host writeable")]
    NotHostWriteable,
}

pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: ManuallyDrop<Allocation>,
    pub(crate) location: MemoryLocation,
    pub(crate) usage: vk::BufferUsageFlags,
    pub(crate) size: u64,
    pub(crate) allocator: Arc<Mutex<Allocator>>,
    pub(crate) device: Arc<Device>,
}

impl Buffer {
    pub fn new(
        device: &Arc<Device>,
        allocator: &Arc<Mutex<Allocator>>,
        location: MemoryLocation,
        usage: vk::BufferUsageFlags,
        size: u64,
    ) -> Result<Self, BufferError> {
        let buffer = unsafe {
            device
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .map_err(BufferError::CreateError)?
        };
        let mem_req = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let memory = allocator
            .lock()
            .map_err(|e| BufferError::AllocationError(AllocError::LockError(format!("{e}"))))?
            .allocate(&AllocationCreateDesc {
                name: &format!("{:x}", buffer.as_raw()),
                requirements: mem_req,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(AllocError::LibError)
            .map_err(BufferError::AllocationError)?;
        unsafe {
            device
                .device
                .bind_buffer_memory(buffer, memory.memory(), memory.offset())
                .map_err(BufferError::MemoryBindError)?;
        }
        Ok(Self {
            buffer,
            memory: ManuallyDrop::new(memory),
            location,
            usage,
            size,
            allocator: allocator.clone(),
            device: device.clone(),
        })
    }

    pub fn write_data(&mut self, offset: u64, data: &[u8]) -> Result<(), BufferError> {
        let mapped_slice = self
            .memory
            .mapped_slice_mut()
            .ok_or(BufferError::NotHostWriteable)?;
        let write_range = &mut mapped_slice[offset as usize..];
        let inp_bytes = &data[..write_range.len()];
        write_range.copy_from_slice(inp_bytes);
        Ok(())
    }

    pub fn new_c2g_with_data<'a>(
        device: &Arc<Device>,
        allocator: &Arc<Mutex<Allocator>>,
        usage: vk::BufferUsageFlags,
        data: &[u8],
    ) -> Result<Self, BufferError> {
        let mut buffer = Self::new(
            device,
            allocator,
            MemoryLocation::CpuToGpu,
            usage,
            data.len() as _,
        )?;
        buffer.write_data(0, data)?;
        Ok(buffer)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
        }
    }
}
