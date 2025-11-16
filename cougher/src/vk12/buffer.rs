use std::mem::ManuallyDrop;

use ash::vk::{self, Handle};
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

use crate::make_init_struct_copy;

make_init_struct_copy!(
    InitBuffer,
    vk::Buffer,
    self,
    self.device.destroy_buffer(self.inner, None)
);

#[derive(Debug, thiserror::Error)]
pub enum BufferError {
    #[error("Error creating Vulkan Buffer: {0}")]
    CreateError(vk::Result),
    #[error("Error allocation memory for  Vulkan Image: {0}")]
    AllocationError(AllocationError),
    #[error("Buffer is not host writeable")]
    NotHostWriteable,
}

pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) size: u64,
    pub(crate) allocation: ManuallyDrop<Allocation>,
    needs_cleanup: bool,
}

impl Buffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        location: MemoryLocation,
        size: u64,
    ) -> Result<Self, BufferError> {
        let buffer = unsafe {
            device
                .create_buffer(&vk::BufferCreateInfo::default().size(size), None)
                .map_err(BufferError::CreateError)?
        };
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: &format!("{:x}", buffer.as_raw()),
                requirements: mem_req,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map(ManuallyDrop::new)
            .map_err(BufferError::AllocationError)?;
        Ok(Self {
            buffer,
            size,
            allocation,
            needs_cleanup: true,
        })
    }

    pub fn write_data(&mut self, offset: u64, data: &[u8]) -> Result<(), BufferError> {
        let mapped_slice = self
            .allocation
            .mapped_slice_mut()
            .ok_or(BufferError::NotHostWriteable)?;
        let write_range = &mut mapped_slice[offset as usize..];
        let inp_bytes = &data[..write_range.len()];
        write_range.copy_from_slice(inp_bytes);
        Ok(())
    }

    pub fn new_cpu_to_gpu_with_data(
        device: &ash::Device,
        allocator: &mut Allocator,
        data: &[u8],
    ) -> Result<Self, BufferError> {
        let mut buffer = Self::new(device, allocator, MemoryLocation::CpuToGpu, data.len() as _)?;
        buffer.write_data(0, data)?;
        Ok(buffer)
    }

    pub fn cleanup(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if self.needs_cleanup {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                self.needs_cleanup = false;
                let _ = allocator
                    .free(ManuallyDrop::take(&mut self.allocation))
                    .inspect_err(|e| eprintln!("warning: error cleaning up gpu allocation: {e}"));
            }
        }
    }
}
