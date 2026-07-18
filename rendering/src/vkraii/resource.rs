use std::{
    mem::ManuallyDrop,
    ops::Range,
    sync::{Arc, Mutex, PoisonError},
};

use anyhow::Context;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use hashbrown::HashMap;

use crate::vkraii::device::DeviceDropper;

pub struct Memory {
    pub allocation: ManuallyDrop<Allocation>,
    allocator: Arc<Mutex<Allocator>>,
}

impl Drop for Memory {
    fn drop(&mut self) {
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        self.allocator
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .free(allocation)
            .inspect_err(|e| log::warn!("freeing memory failed: {e}"))
            .ok();
    }
}

pub struct BufferRaii {
    pub buffer: vk::Buffer,
    pub mem: Memory,
    pub size: u64,
    device_d: Arc<DeviceDropper>,
}

impl BufferRaii {
    pub fn new(
        device_d: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        create_info: &vk::BufferCreateInfo,
        mem_location: MemoryLocation,
    ) -> anyhow::Result<Self> {
        let buffer = unsafe {
            device_d
                .device
                .create_buffer(create_info, None)
                .with_context(|| "creating vk buffer failed")?
        };
        let mem_req = unsafe { device_d.device.get_buffer_memory_requirements(buffer) };
        let mut allocator_mut = allocator.lock().unwrap_or_else(PoisonError::into_inner);
        let allocation = allocator_mut
            .allocate(&AllocationCreateDesc {
                name: &format!("buffer_{buffer:?}"),
                requirements: mem_req,
                location: mem_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating memory failed")?;
        unsafe {
            device_d
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .with_context(|| "binding buffer memory failed")?;
        }
        Ok(Self {
            buffer,
            mem: Memory {
                allocation: ManuallyDrop::new(allocation),
                allocator: allocator.clone(),
            },
            size: create_info.size,
            device_d: device_d.clone(),
        })
    }
}

impl Drop for BufferRaii {
    fn drop(&mut self) {
        unsafe {
            self.device_d.device.destroy_buffer(self.buffer, None);
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewKey {
    pub type_: vk::ImageViewType,
    pub layer_range: Range<u32>,
    pub level_range: Range<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageAccess {
    pub access_flags: vk::AccessFlags,
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags,
}

pub struct ImageRaii {
    pub image: vk::Image,
    pub memory: Option<Memory>,
    pub res: (u32, u32, u32),
    pub format: vk::Format,
    pub layers: u32,
    pub levels: u32,
    pub access: ImageAccess,
    pub views: HashMap<ImageViewKey, vk::ImageView>,
    pub device_d: Arc<DeviceDropper>,
}

impl ImageRaii {
    pub fn new(
        device_d: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        create_info: &vk::ImageCreateInfo,
        mem_location: MemoryLocation,
    ) -> anyhow::Result<Self> {
        let image = unsafe {
            device_d
                .device
                .create_image(create_info, None)
                .with_context(|| "creating vk image failed")?
        };
        let mem_req = unsafe { device_d.device.get_image_memory_requirements(image) };
        let mut allocator_mut = allocator.lock().unwrap_or_else(PoisonError::into_inner);
        let allocation = allocator_mut
            .allocate(&AllocationCreateDesc {
                name: &format!("image_{image:?}"),
                requirements: mem_req,
                location: mem_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating memory failed")?;
        unsafe {
            device_d
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .with_context(|| "binding image memory failed")?;
        }
        Ok(Self {
            image,
            memory: Some(Memory {
                allocation: ManuallyDrop::new(allocation),
                allocator: allocator.clone(),
            }),
            res: (
                create_info.extent.width,
                create_info.extent.height,
                create_info.extent.depth,
            ),
            format: create_info.format,
            layers: create_info.array_layers,
            levels: create_info.mip_levels,
            access: ImageAccess {
                access_flags: vk::AccessFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
                stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            },
            views: Default::default(),
            device_d: device_d.clone(),
        })
    }

    pub fn get_view(&mut self, key: &ImageViewKey) -> anyhow::Result<vk::ImageView> {
        let view = self.views.get(key).cloned();
        let view = match view {
            Some(t) => t,
            None => {
                let view = unsafe {
                    self.device_d
                        .device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .format(self.format)
                                .image(self.image)
                                .subresource_range(self.subresource_range(
                                    key.layer_range.clone(),
                                    key.level_range.clone(),
                                ))
                                .view_type(key.type_),
                            None,
                        )
                        .with_context(|| "vk image view creation failed")?
                };
                self.views.insert(key.clone(), view);
                view
            }
        };
        Ok(view)
    }

    pub fn subresource_range(
        &self,
        layer_range: Range<u32>,
        level_range: Range<u32>,
    ) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(image_aspect_mask(self.format))
            .base_array_layer(layer_range.start)
            .base_mip_level(level_range.start)
            .layer_count(layer_range.end - layer_range.start)
            .level_count(level_range.end - level_range.start)
    }

    pub fn subresource_layers(
        &self,
        layer_range: Range<u32>,
        level: u32,
    ) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(image_aspect_mask(self.format))
            .base_array_layer(layer_range.start)
            .layer_count(layer_range.end - layer_range.start)
            .mip_level(level)
    }

    pub fn barrier(
        &mut self,
        command_buffer: vk::CommandBuffer,
        new_access: ImageAccess,
        layer_range: Range<u32>,
        level_range: Range<u32>,
    ) {
        if self.access == new_access {
            return;
        }
        unsafe {
            self.device_d.device.cmd_pipeline_barrier(
                command_buffer,
                self.access.stage,
                new_access.stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(new_access.access_flags)
                    .dst_queue_family_index(self.device_d.graphics_qf)
                    .image(self.image)
                    .new_layout(new_access.layout)
                    .old_layout(self.access.layout)
                    .src_access_mask(self.access.access_flags)
                    .src_queue_family_index(self.device_d.graphics_qf)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(image_aspect_mask(self.format))
                            .base_array_layer(layer_range.start)
                            .base_mip_level(level_range.start)
                            .layer_count(layer_range.end - layer_range.start)
                            .level_count(level_range.end - level_range.start),
                    )],
            );
        }
        self.access = new_access;
    }
}

impl Drop for ImageRaii {
    fn drop(&mut self) {
        for (_, view) in self.views.drain() {
            unsafe {
                self.device_d.device.destroy_image_view(view, None);
            }
        }
        if self.memory.is_some() {
            unsafe {
                self.device_d.device.destroy_image(self.image, None);
            }
        }
    }
}

pub fn is_depth_stencil(fmt: vk::Format) -> (bool, bool) {
    match fmt {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            (true, false)
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => (true, true),
        _ => (false, false),
    }
}

pub fn image_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
    let (is_d, has_s) = is_depth_stencil(fmt);
    if is_d {
        if has_s {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}
