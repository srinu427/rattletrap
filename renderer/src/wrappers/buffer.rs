use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use thiserror::Error;

use crate::wrappers::{
    gpu_allocation::{GpuAllocation, GpuAllocationError},
    logical_device::LogicalDevice,
};

#[derive(Debug, Error)]
pub enum BufferError {
    #[error("Error creating buffer: {0}")]
    CreationError(vk::Result),
    #[error("GPU Allocation error: {0}")]
    AllocationError(#[from] GpuAllocationError),
    #[error("Error binding memory to buffer: {0}")]
    MemoryBindError(vk::Result),
    #[error("No allocation found for buffer")]
    NoAllocationError,
    #[error("No CPU mapping found for buffer")]
    NoCpuMappingError,
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Buffer {
    #[get_copy = "pub"]
    buffer: vk::Buffer,
    #[get = "pub"]
    allocation: Option<GpuAllocation>,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl Buffer {
    pub fn new(
        device: Arc<LogicalDevice>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, BufferError> {
        let buffer_create_info = vk::BufferCreateInfo::default().size(size).usage(usage);

        let buffer = unsafe {
            device
                .device()
                .create_buffer(&buffer_create_info, None)
                .map_err(BufferError::CreationError)?
        };

        Ok(Self {
            buffer,
            allocation: None,
            device,
        })
    }

    pub fn allocate_memory(
        &mut self,
        allocator: Arc<Mutex<Allocator>>,
        gpu_only: bool,
    ) -> Result<(), BufferError> {
        let requirements = unsafe {
            self.device
                .device()
                .get_buffer_memory_requirements(self.buffer)
        };
        let mem_location = if gpu_only {
            gpu_allocator::MemoryLocation::GpuOnly
        } else {
            gpu_allocator::MemoryLocation::CpuToGpu
        };
        let allocation = GpuAllocation::new(
            allocator,
            &format!("buffer_{:?}", self.buffer),
            requirements,
            true,
            mem_location,
        )?;
        unsafe {
            self.device
                .device()
                .bind_buffer_memory(
                    self.buffer,
                    allocation.allocation().memory(),
                    allocation.allocation().offset(),
                )
                .map_err(BufferError::MemoryBindError)?
        };
        self.allocation = Some(allocation);
        Ok(())
    }

    pub fn get_allocation_mount_slice(&mut self) -> Result<&mut [u8], BufferError> {
        self.allocation
            .as_mut()
            .ok_or(BufferError::NoAllocationError)?
            .allocation_mut()
            .mapped_slice_mut()
            .ok_or(BufferError::NoCpuMappingError)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_buffer(self.buffer, None);
        }
    }
}
