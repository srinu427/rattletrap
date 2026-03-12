use std::sync::Arc;

use ash::vk::{self, Handle};
use rhi2::enumflags2::BitFlags;

use crate::{device::DeviceDropper, memory::Memory};

fn rhi2_buf_flags_to_vk_flags(flags: &BitFlags<rhi2::buffer::BufferFlags>) -> vk::BufferUsageFlags {
    let mut vk_flags = vk::BufferUsageFlags::empty();
    for flag in flags.iter() {
        match flag {
            rhi2::buffer::BufferFlags::CopyDst => vk_flags |= vk::BufferUsageFlags::TRANSFER_DST,
            rhi2::buffer::BufferFlags::CopySrc => vk_flags |= vk::BufferUsageFlags::TRANSFER_SRC,
            rhi2::buffer::BufferFlags::Vertex => vk_flags |= vk::BufferUsageFlags::VERTEX_BUFFER,
            rhi2::buffer::BufferFlags::Index => vk_flags |= vk::BufferUsageFlags::INDEX_BUFFER,
        }
    }
    vk_flags
}

pub struct BufferDropper {
    pub handle: vk::Buffer,
    pub memory: Memory,
    pub device_dropper: Arc<DeviceDropper>,
}

impl BufferDropper {
    pub fn new(
        device_dropper: &Arc<DeviceDropper>,
        size: u64,
        flags: &BitFlags<rhi2::buffer::BufferFlags>,
        host_access: rhi2::HostAccess,
    ) -> Result<Self, String> {
        let usage = rhi2_buf_flags_to_vk_flags(flags);
        let create_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        let handle = unsafe {
            device_dropper
                .device
                .create_buffer(&create_info, None)
                .map_err(|e| format!("vk buffer creation failed: {e}"))?
        };
        let reqs = unsafe { device_dropper.device.get_buffer_memory_requirements(handle) };
        let memory = Memory::new(
            &device_dropper.allocator,
            reqs,
            &format!("{:x}", handle.as_raw()),
            host_access,
        )
        .map_err(|e| format!("buffer mem allocation failed: {e}"))?;
        Ok(Self {
            handle,
            memory,
            device_dropper: device_dropper.clone(),
        })
    }
}

impl Drop for BufferDropper {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper.device.destroy_buffer(self.handle, None);
        }
    }
}

#[derive(Clone)]
pub struct Buffer {
    dropper: Arc<BufferDropper>,
    size: usize,
    usage: BitFlags<rhi2::buffer::BufferFlags>,
    host_access: rhi2::HostAccess,
}

impl rhi2::buffer::Buffer for Buffer {
    fn size(&self) -> usize {
        self.size
    }

    fn host_access(&self) -> rhi2::HostAccess {
        self.host_access
    }

    fn host_write(&self, data: &[u8]) -> Result<(), rhi2::buffer::BufferErr> {
        let mem_ptr = self.dropper.memory.handle.mapped_slice_mut();
        todo!()
    }
}
