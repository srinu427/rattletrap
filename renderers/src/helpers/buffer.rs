use std::{marker::PhantomData, mem};

use anyhow::Context;
use ash::vk;
use bytemuck::NoUninit;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::helpers::{
    device::{GpuCommandRecorder, GpuCtx},
    free_allocation_logged,
};

fn create_buffer(
    ctx: &mut GpuCtx,
    mut usage: vk::BufferUsageFlags,
    size: u64,
    cpu_write: bool,
) -> anyhow::Result<(vk::Buffer, Allocation)> {
    usage |= vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let buffer = unsafe {
        ctx.device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(usage),
            None,
        )?
    };
    let requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
    let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
        name: &format!("buffer_{:?}", buffer),
        requirements,
        location: if cpu_write {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        },
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    unsafe {
        ctx.device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
    }
    Ok((buffer, allocation))
}

fn copy_b2b(
    ctx: &mut GpuCtx,
    cr: &mut GpuCommandRecorder,
    src: vk::Buffer,
    src_offset: u64,
    dst: vk::Buffer,
    dst_offset: u64,
    len: u64,
) -> anyhow::Result<()> {
    cr.begin(ctx)?;
    unsafe {
        ctx.device.cmd_copy_buffer(
            cr.cb,
            src,
            dst,
            &[vk::BufferCopy {
                src_offset,
                dst_offset,
                size: len,
            }],
        );
        ctx.device.cmd_pipeline_barrier(
            cr.cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[vk::BufferMemoryBarrier::default()
                .buffer(dst)
                .dst_access_mask(vk::AccessFlags::empty())
                .dst_queue_family_index(ctx.graphics_q.family)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(ctx.graphics_q.family)],
            &[],
        );
    }
    Ok(())
}

pub struct GpuVec<T: NoUninit> {
    pub buffer: vk::Buffer,
    allocation: Option<Allocation>,
    capacity: usize,
    pub len: usize,
    usage: vk::BufferUsageFlags,
    _phantom: PhantomData<T>,
}

impl<T: NoUninit> GpuVec<T> {
    pub fn new(ctx: &mut GpuCtx, usage: vk::BufferUsageFlags) -> anyhow::Result<Self> {
        Self::with_capacity(ctx, usage, 1)
    }

    pub fn with_capacity(
        ctx: &mut GpuCtx,
        usage: vk::BufferUsageFlags,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let buf_size = capacity as u64 * size_of::<T>() as u64;
        let (buffer, allocation) = create_buffer(ctx, usage, buf_size, ctx.gpu_info.unified_mem)?;
        Ok(Self {
            buffer,
            allocation: Some(allocation),
            capacity,
            len: 0,
            usage,
            _phantom: Default::default(),
        })
    }

    fn grow(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        atleast: usize,
    ) -> anyhow::Result<()> {
        let old_buf_size = (self.len * size_of::<T>()) as u64;
        let new_capacity = atleast.next_power_of_two();
        let new_buf_size = (new_capacity * size_of::<T>()) as u64;
        let (new_buffer, new_allocation) =
            create_buffer(ctx, self.usage, new_buf_size, ctx.gpu_info.unified_mem)?;
        if self.len > 0 {
            copy_b2b(ctx, cr, self.buffer, 0, new_buffer, 0, old_buf_size)?;
            let old_buffer = mem::replace(&mut self.buffer, new_buffer);
            let old_allocation = self.allocation.replace(new_allocation);
            if let Some(old_allocation) = old_allocation {
                cr.preserve_buffers.push((old_buffer, old_allocation));
            }
        } else {
            let old_buffer = mem::replace(&mut self.buffer, new_buffer);
            let old_allocation = self.allocation.replace(new_allocation);
            if let Some(old_allocation) = old_allocation {
                unsafe {
                    ctx.device.destroy_buffer(old_buffer, None);
                    free_allocation_logged(&mut ctx.allocator, old_allocation);
                }
            }
        }
        self.capacity = new_capacity;
        Ok(())
    }

    pub fn write(
        &mut self,
        ctx: &mut GpuCtx,
        offset: usize,
        elems: &[T],
        cr: &mut GpuCommandRecorder,
    ) -> anyhow::Result<()> {
        if elems.len() == 0 {
            return Ok(());
        }
        let size = size_of::<T>();
        // Bounds check
        if (offset + elems.len()) > self.capacity {
            self.grow(ctx, cr, offset + elems.len())?;
        }
        let write_offset = offset * size;
        let write_size = elems.len() * size;
        if ctx.gpu_info.unified_mem {
            let mem_slice = self
                .allocation
                .as_mut()
                .with_context(|| "no allocated memory for gpu_vec to write")?
                .mapped_slice_mut()
                .with_context(|| "memory not cpu writeable")?;
            mem_slice[write_offset..(write_offset + write_size)]
                .copy_from_slice(bytemuck::cast_slice(elems));
        } else {
            let (stage_buffer, mut stage_allocation) =
                create_buffer(ctx, vk::BufferUsageFlags::empty(), write_size as u64, true)?;
            let mem_slice = stage_allocation
                .mapped_slice_mut()
                .with_context(|| "staging memory not cpu writeable")?;
            mem_slice[..write_size].copy_from_slice(bytemuck::cast_slice(elems));
            copy_b2b(
                ctx,
                cr,
                stage_buffer,
                0,
                self.buffer,
                write_offset as u64,
                write_size as u64,
            )?;
            cr.preserve_buffers.push((stage_buffer, stage_allocation));
        }
        if (offset + elems.len()) > self.len {
            self.len = offset + elems.len();
        }
        Ok(())
    }

    pub fn push(
        &mut self,
        ctx: &mut GpuCtx,
        elem: &T,
        cr: &mut GpuCommandRecorder,
    ) -> anyhow::Result<()> {
        if self.len == self.capacity {
            self.grow(ctx, cr, self.capacity + 1)?;
        }
        let insert_idx = self.len;
        let size = size_of::<T>();
        let write_offset = insert_idx * size;
        self.len += 1;
        if ctx.gpu_info.unified_mem {
            let mem_slice = self
                .allocation
                .as_mut()
                .with_context(|| "no allocated memory for gpu_vec to push")?
                .mapped_slice_mut()
                .with_context(|| "memory not cpu writeable")?;
            mem_slice[write_offset..(write_offset + size)]
                .copy_from_slice(bytemuck::bytes_of(elem));
        } else {
            let (stage_buffer, mut stage_allocation) =
                create_buffer(ctx, vk::BufferUsageFlags::TRANSFER_SRC, size as u64, true)?;
            let mem_slice = stage_allocation
                .mapped_slice_mut()
                .with_context(|| "staging memory not cpu writeable")?;
            mem_slice[..size].copy_from_slice(bytemuck::bytes_of(elem));
            copy_b2b(
                ctx,
                cr,
                stage_buffer,
                0,
                self.buffer,
                write_offset as u64,
                size as u64,
            )?;
            cr.preserve_buffers.push((stage_buffer, stage_allocation));
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        if let Some(altn) = self.allocation.take() {
            unsafe {
                ctx.device.destroy_buffer(self.buffer, None);
            }
            if let Err(e) = ctx.allocator.free(altn) {
                log::warn!("freeing memory of gpu vector {:?} failed: {e}", self.buffer)
            };
        }
    }
}

impl<T: NoUninit> Drop for GpuVec<T> {
    fn drop(&mut self) {
        if self.allocation.is_some() {
            log::error!("gpu_vec {:?} is not destroyed properly", self.buffer);
        }
    }
}

pub struct GpuObj<T: NoUninit> {
    pub buffer: vk::Buffer,
    allocation: Option<Allocation>,
    _phantom: PhantomData<T>,
}

impl<T: NoUninit> GpuObj<T> {
    pub fn new(ctx: &mut GpuCtx, usage: vk::BufferUsageFlags) -> anyhow::Result<Self> {
        let buf_size = size_of::<T>() as u64;
        let (buffer, allocation) = create_buffer(ctx, usage, buf_size, ctx.gpu_info.unified_mem)?;
        Ok(Self {
            buffer,
            allocation: Some(allocation),
            _phantom: Default::default(),
        })
    }

    pub fn with_data(
        ctx: &mut GpuCtx,
        usage: vk::BufferUsageFlags,
        cr: &mut GpuCommandRecorder,
        data: &T,
    ) -> anyhow::Result<Self> {
        let mut out = Self::new(ctx, usage)?;
        out.write(ctx, data, cr)?;
        Ok(out)
    }

    pub fn write(
        &mut self,
        ctx: &mut GpuCtx,
        elem: &T,
        cr: &mut GpuCommandRecorder,
    ) -> anyhow::Result<()> {
        let size = size_of::<T>();
        if ctx.gpu_info.unified_mem {
            let mem_slice = self
                .allocation
                .as_mut()
                .with_context(|| "no allocated memory for gpu_vec to push")?
                .mapped_slice_mut()
                .with_context(|| "memory not cpu writeable")?;
            mem_slice[0..size].copy_from_slice(bytemuck::bytes_of(elem));
        } else {
            let (stage_buffer, mut stage_allocation) =
                create_buffer(ctx, vk::BufferUsageFlags::empty(), size as u64, true)?;
            let mem_slice = stage_allocation
                .mapped_slice_mut()
                .with_context(|| "staging memory not cpu writeable")?;
            mem_slice[..size].copy_from_slice(bytemuck::bytes_of(elem));
            copy_b2b(ctx, cr, stage_buffer, 0, self.buffer, 0, size as u64)?;
            cr.preserve_buffers.push((stage_buffer, stage_allocation));
        }
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        if let Some(altn) = self.allocation.take() {
            unsafe {
                ctx.device.destroy_buffer(self.buffer, None);
            }
            if let Err(e) = ctx.allocator.free(altn) {
                log::warn!("freeing memory of gpu object {:?} failed: {e}", self.buffer)
            };
        }
    }
}

impl<T: NoUninit> Drop for GpuObj<T> {
    fn drop(&mut self) {
        if self.allocation.is_some() {
            log::error!("gpu_obj {:?} is not destroyed properly", self.buffer);
        }
    }
}
