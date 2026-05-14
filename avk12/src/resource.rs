use std::{
    mem::ManuallyDrop,
    ops::Range,
    sync::{Arc, Mutex, PoisonError, RwLock},
};

use anyhow::Context;
use ash::vk;
use getset::{Setters, WithSetters};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use hashbrown::HashMap;
use typed_builder::TypedBuilder;

use crate::device::DeviceDropper;

pub struct GMem {
    allocation: ManuallyDrop<Allocation>,
    allocator: Arc<Mutex<Allocator>>,
}

impl GMem {
    pub(crate) fn new(
        allocator: &Arc<Mutex<Allocator>>,
        name: &str,
        location: MemoryLocation,
        requirements: vk::MemoryRequirements,
    ) -> anyhow::Result<Self> {
        let allocation = allocator
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .context("allocating memory failed")?;
        Ok(Self {
            allocation: ManuallyDrop::new(allocation),
            allocator: allocator.clone(),
        })
    }
}

impl Drop for GMem {
    fn drop(&mut self) {
        unsafe {
            let altn = ManuallyDrop::take(&mut self.allocation);
            if let Err(e) = self
                .allocator
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .free(altn)
            {
                log::warn!("issue freeing memory: {e}");
            };
        }
    }
}

#[derive(Debug, Clone, WithSetters)]
pub struct BufferCreateInfo {
    #[getset(set_with = "pub")]
    pub size: u64,
    #[getset(set_with = "pub")]
    pub used_for: vk::BufferUsageFlags,
    #[getset(set_with = "pub")]
    pub mem_location: MemoryLocation,
}

impl Default for BufferCreateInfo {
    fn default() -> Self {
        Self {
            size: Default::default(),
            used_for: Default::default(),
            mem_location: MemoryLocation::GpuOnly,
        }
    }
}

pub(crate) struct BufferDropper {
    info: BufferCreateInfo,
    pub(crate) handle: vk::Buffer,
    pub(crate) mem: Mutex<GMem>,
    dd: Arc<DeviceDropper>,
}

impl Drop for BufferDropper {
    fn drop(&mut self) {
        unsafe {
            self.dd.device.destroy_buffer(self.handle, None);
        }
    }
}

#[derive(Clone)]
pub struct BufferRef {
    pub(crate) dropper: Arc<BufferDropper>,
}

impl BufferRef {
    pub(crate) fn new(
        dd: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        info: BufferCreateInfo,
    ) -> anyhow::Result<Self> {
        let handle = unsafe {
            dd.device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(info.size)
                        .usage(info.used_for),
                    None,
                )
                .context("vk buffer creation failed")?
        };
        let req = unsafe { dd.device.get_buffer_memory_requirements(handle) };
        let mem = GMem::new(allocator, &format!("{handle:?}"), info.mem_location, req)?;
        unsafe {
            dd.device
                .bind_buffer_memory(handle, mem.allocation.memory(), mem.allocation.offset())
                .context("binding buffer to memory failed")?;
        }
        Ok(Self {
            dropper: Arc::new(BufferDropper {
                handle,
                mem: Mutex::new(mem),
                dd: dd.clone(),
                info,
            }),
        })
    }

    pub fn len(&self) -> u64 {
        self.dropper.info.size
    }

    pub fn write_cpu(&self, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        match self.dropper.info.mem_location {
            MemoryLocation::CpuToGpu => {}
            _ => {
                return Err(anyhow::Error::msg(
                    "cant write directly on buffers that are not created with DataMoveDir as Cpu2Gpu",
                ));
            }
        };
        let mut mem_mut = self
            .dropper
            .mem
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mapped_mem = mem_mut
            .allocation
            .mapped_slice_mut()
            .context("memory is not cpu accessible")?;
        let dst_mem = &mut mapped_mem[offset as usize..];
        let copy_size = dst_mem.len().min(data.len());
        dst_mem[..copy_size].copy_from_slice(&data[..copy_size]);
        Ok(())
    }

    pub fn full_slice(&self) -> BufferSlice<'_> {
        BufferSlice {
            buffer: self,
            range: Range {
                start: 0,
                end: self.len(),
            },
        }
    }

    pub fn slice(&self, range: Range<u64>) -> BufferSlice<'_> {
        BufferSlice {
            buffer: self,
            range,
        }
    }
}

pub struct BufferSlice<'a> {
    pub buffer: &'a BufferRef,
    pub range: Range<u64>,
}

pub(crate) trait FormatMeta {
    fn is_depth(&self) -> bool;
    fn has_stencil(&self) -> bool;
    fn rem_srgb(&self) -> Self;
    fn aspect_flag(&self) -> vk::ImageAspectFlags;
}

impl FormatMeta for vk::Format {
    fn is_depth(&self) -> bool {
        match *self {
            Self::D16_UNORM
            | Self::D16_UNORM_S8_UINT
            | Self::D24_UNORM_S8_UINT
            | Self::D32_SFLOAT
            | Self::D32_SFLOAT_S8_UINT
            | Self::X8_D24_UNORM_PACK32 => true,
            _ => false,
        }
    }

    fn has_stencil(&self) -> bool {
        match *self {
            Self::D16_UNORM_S8_UINT | Self::D24_UNORM_S8_UINT | Self::D32_SFLOAT_S8_UINT => true,
            _ => false,
        }
    }

    fn rem_srgb(&self) -> Self {
        match *self {
            Self::R8_SRGB => Self::R8_UNORM,
            Self::R8G8_SRGB => Self::R8G8_UINT,
            Self::B8G8R8_SRGB => Self::B8G8R8_UINT,
            Self::R8G8B8_SRGB => Self::R8G8B8_UINT,
            Self::B8G8R8A8_SRGB => Self::B8G8R8A8_UINT,
            Self::R8G8B8A8_SRGB => Self::R8G8B8A8_UINT,
            Self::A8B8G8R8_SRGB_PACK32 => Self::A8B8G8R8_UINT_PACK32,
            _ => self.clone(),
        }
    }
    fn aspect_flag(&self) -> vk::ImageAspectFlags {
        match *self {
            Self::D16_UNORM | Self::D32_SFLOAT | Self::X8_D24_UNORM_PACK32 => {
                vk::ImageAspectFlags::DEPTH
            }
            Self::D16_UNORM_S8_UINT | Self::D24_UNORM_S8_UINT | Self::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::COLOR,
        }
    }
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct ImageCreateInfo {
    pub format: vk::Format,
    pub res: (u32, u32, u32),
    #[builder(default = 1)]
    pub layers: u32,
    #[builder(default = 1)]
    pub mip_levels: u32,
    #[builder(default)]
    pub used_for: vk::ImageUsageFlags,
    #[builder(default=MemoryLocation::GpuOnly)]
    pub mem_location: MemoryLocation,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub layer_range: Range<u32>,
    pub level_range: Range<u32>,
}

#[derive(Debug, Clone)]
pub struct ImageAccess {
    pub layout: vk::ImageLayout,
    pub access_flags: vk::AccessFlags,
    pub access_stage: vk::PipelineStageFlags,
}

pub(crate) struct ImageDropper {
    pub(crate) info: ImageCreateInfo,
    pub(crate) handle: vk::Image,
    pub(crate) mem: Option<GMem>,
    pub(crate) dont_drop: bool,
    pub(crate) dd: Arc<DeviceDropper>,
    pub(crate) view_cache: RwLock<HashMap<ImageViewInfo, vk::ImageView>>,
    pub(crate) last_access: RwLock<ImageAccess>,
}

impl Drop for ImageDropper {
    fn drop(&mut self) {
        unsafe {
            for (_, view) in self
                .view_cache
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .drain()
            {
                self.dd.device.destroy_image_view(view, None);
            }
            if !self.dont_drop {
                self.dd.device.destroy_image(self.handle, None);
            }
        }
    }
}

#[derive(Clone)]
pub struct ImageRef {
    pub(crate) dropper: Arc<ImageDropper>,
}

impl ImageRef {
    pub(crate) fn new(
        dd: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        info: ImageCreateInfo,
    ) -> anyhow::Result<Self> {
        let image_type = if info.res.2 == 1 {
            vk::ImageType::TYPE_2D
        } else {
            vk::ImageType::TYPE_3D
        };
        let handle = unsafe {
            dd.device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(image_type)
                        .format(info.format)
                        .extent(vk::Extent3D {
                            width: info.res.0,
                            height: info.res.1,
                            depth: info.res.2,
                        })
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .mip_levels(info.mip_levels)
                        .array_layers(info.layers)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(info.used_for),
                    None,
                )
                .context("image creation failed")?
        };
        let req = unsafe { dd.device.get_image_memory_requirements(handle) };
        let mem = GMem::new(allocator, &format!("{handle:?}"), info.mem_location, req)?;
        unsafe {
            dd.device
                .bind_image_memory(handle, mem.allocation.memory(), mem.allocation.offset())
                .context("binding buffer to memory failed")?
        }
        Ok(Self {
            dropper: Arc::new(ImageDropper {
                info,
                handle,
                mem: Some(mem),
                dont_drop: false,
                dd: dd.clone(),
                view_cache: RwLock::new(HashMap::new()),
                last_access: RwLock::new(ImageAccess {
                    layout: vk::ImageLayout::UNDEFINED,
                    access_flags: vk::AccessFlags::empty(),
                    access_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                }),
            }),
        })
    }

    fn make_view(&self, key: &ImageViewInfo) -> anyhow::Result<vk::ImageView> {
        let view = unsafe {
            self.dropper
                .dd
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(self.dropper.handle)
                        .format(self.dropper.info.format)
                        .components(vk::ComponentMapping::default())
                        .view_type(key.view_type)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(self.dropper.info.format.aspect_flag())
                                .base_array_layer(key.layer_range.start)
                                .base_mip_level(key.level_range.start)
                                .layer_count(key.layer_range.end - key.layer_range.start)
                                .level_count(key.level_range.end - key.level_range.start),
                        ),
                    None,
                )
                .context("image view creation failed")?
        };
        Ok(view)
    }

    pub fn view(&self, info: &ImageViewInfo) -> anyhow::Result<ImageView> {
        let ex_view = self
            .dropper
            .view_cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(info)
            .cloned();
        let handle = match ex_view {
            Some(h) => h,
            None => {
                let view = self.make_view(info)?;
                self.dropper
                    .view_cache
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(info.clone(), view);
                view
            }
        };
        Ok(ImageView {
            image_droppper: self.dropper.clone(),
            handle,
            info: info.clone(),
        })
    }
}

#[derive(Clone)]
pub struct ImageView {
    pub(crate) image_droppper: Arc<ImageDropper>,
    pub(crate) handle: vk::ImageView,
    pub info: ImageViewInfo,
}

pub(crate) struct SamplerDropper {
    pub(crate) handle: vk::Sampler,
    dd: Arc<DeviceDropper>,
}

impl Drop for SamplerDropper {
    fn drop(&mut self) {
        unsafe {
            self.dd.device.destroy_sampler(self.handle, None);
        }
    }
}

#[derive(Clone)]
pub struct Sampler {
    pub(crate) dropper: Arc<SamplerDropper>,
}

impl Sampler {
    pub(crate) fn new(dd: &Arc<DeviceDropper>) -> anyhow::Result<Self> {
        let handle = unsafe {
            dd.device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .context("sampler creation failed")?
        };
        Ok(Self {
            dropper: Arc::new(SamplerDropper {
                handle,
                dd: dd.clone(),
            }),
        })
    }
}
