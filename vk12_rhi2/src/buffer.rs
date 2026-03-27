use std::sync::Arc;

use ash::vk::{self, Handle};
use rhi2::{buffer::BufferFlags, enumflags2::BitFlags};

use crate::{
    device::DeviceDropper,
    memory::{MemAlloc, Memory},
};

fn rhi2_buf_flags_to_vk_flags(flags: &BitFlags<BufferFlags>) -> vk::BufferUsageFlags {
    let mut vk_flags = vk::BufferUsageFlags::empty();
    for flag in flags.iter() {
        match flag {
            BufferFlags::CopyDst => vk_flags |= vk::BufferUsageFlags::TRANSFER_DST,
            BufferFlags::CopySrc => vk_flags |= vk::BufferUsageFlags::TRANSFER_SRC,
            BufferFlags::Vertex => vk_flags |= vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferFlags::Index => vk_flags |= vk::BufferUsageFlags::INDEX_BUFFER,
            BufferFlags::Uniform => vk_flags |= vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferFlags::Storage => vk_flags |= vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }
    vk_flags
}

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: Memory,
    pub size: usize,
    pub flags: BitFlags<rhi2::buffer::BufferFlags>,
    pub host_access: rhi2::HostAccess,
    pub device_dropper: Arc<DeviceDropper>,
}

impl Buffer {
    pub fn new(
        allocator: &Arc<MemAlloc>,
        size: usize,
        flags: BitFlags<rhi2::buffer::BufferFlags>,
        host_access: rhi2::HostAccess,
    ) -> Result<Self, String> {
        let usage = rhi2_buf_flags_to_vk_flags(&flags);
        let create_info = vk::BufferCreateInfo::default().size(size as _).usage(usage);
        let handle = unsafe {
            allocator
                .device_dropper
                .device
                .create_buffer(&create_info, None)
                .map_err(|e| format!("vk buffer creation failed: {e}"))?
        };
        let reqs = unsafe {
            allocator
                .device_dropper
                .device
                .get_buffer_memory_requirements(handle)
        };
        let memory = Memory::new(
            allocator,
            reqs,
            &format!("{:x}", handle.as_raw()),
            host_access,
        )
        .map_err(|e| format!("mem allocation failed: {e}"))?;
        unsafe {
            allocator
                .device_dropper
                .device
                .bind_buffer_memory(handle, memory.handle.memory(), memory.handle.offset())
                .map_err(|e| format!("bind mem to buffer failed: {e}"))?;
        }
        Ok(Self {
            handle,
            memory,
            device_dropper: allocator.device_dropper.clone(),
            size: size,
            flags,
            host_access,
        })
    }
}

impl rhi2::buffer::Buffer for Buffer {
    fn size(&self) -> usize {
        self.size
    }

    fn host_access(&self) -> rhi2::HostAccess {
        self.host_access
    }

    fn host_write(&mut self, offset: usize, data: &[u8]) -> Result<(), rhi2::buffer::BufferErr> {
        match self.memory.handle.mapped_slice_mut() {
            Some(mem_mut) => {
                let mem_mut = &mut mem_mut[offset..];
                let copy_size = mem_mut.len().min(data.len());
                if copy_size != 0 {
                    mem_mut[..copy_size].copy_from_slice(&data[..copy_size]);
                }
                Ok(())
            }
            None => Err(rhi2::buffer::BufferErr::NotHostWriteable),
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper.device.destroy_buffer(self.handle, None);
        }
    }
}
