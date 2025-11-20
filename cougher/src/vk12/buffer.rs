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
    #[error("Error binding memeory to Buffer: {0}")]
    MemoryBindError(vk::Result),
    #[error("Buffer is not host writeable")]
    NotHostWriteable,
}

pub fn new_buffer<'a>(
    device: &'a ash::Device,
    allocator: &'_ mut Allocator,
    location: MemoryLocation,
    usage: vk::BufferUsageFlags,
    size: u64,
) -> Result<(InitBuffer<'a>, Allocation), BufferError> {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage),
                None,
            )
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
        .map_err(BufferError::AllocationError)?;
    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(BufferError::MemoryBindError)?;
    }
    let init_buffer = InitBuffer {
        drop: true,
        inner: buffer,
        device,
    };
    Ok((init_buffer, allocation))
}

pub fn write_data(
    allocation: &mut Allocation,
    offset: u64,
    data: &[u8],
) -> Result<(), BufferError> {
    let mapped_slice = allocation
        .mapped_slice_mut()
        .ok_or(BufferError::NotHostWriteable)?;
    let write_range = &mut mapped_slice[offset as usize..];
    let inp_bytes = &data[..write_range.len()];
    write_range.copy_from_slice(inp_bytes);
    Ok(())
}

pub fn new_c2g_buffer_with_data<'a>(
    device: &'a ash::Device,
    allocator: &'_ mut Allocator,
    usage: vk::BufferUsageFlags,
    data: &[u8],
) -> Result<(InitBuffer<'a>, Allocation), BufferError> {
    let (buffer, mut allocation) = new_buffer(
        device,
        allocator,
        MemoryLocation::CpuToGpu,
        usage,
        data.len() as _,
    )?;
    write_data(&mut allocation, 0, data)?;
    Ok((buffer, allocation))
}
