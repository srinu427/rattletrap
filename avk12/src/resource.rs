use std::{
    mem::ManuallyDrop,
    ops::Range,
    sync::{Arc, Mutex, PoisonError, RwLock},
};

use anyhow::Context;
use ash::vk;
use enumflags2::{BitFlags, bitflags};
use getset::Getters;
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
        data_dir: DataMoveDir,
        requirements: vk::MemoryRequirements,
    ) -> anyhow::Result<Self> {
        let location = match data_dir {
            DataMoveDir::None => MemoryLocation::GpuOnly,
            DataMoveDir::Cpu2Gpu => MemoryLocation::CpuToGpu,
            DataMoveDir::Gpu2Cpu => MemoryLocation::GpuToCpu,
        };
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

#[derive(Debug, Clone, Copy)]
pub enum DataMoveDir {
    None,
    Cpu2Gpu,
    Gpu2Cpu,
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BufferUsageFlag {
    CopyDst,
    CopySrc,
    Vertex,
    Index,
    Uniform,
    Storage,
}

impl BufferUsageFlag {
    pub(crate) fn to_vk(&self) -> vk::BufferUsageFlags {
        match self {
            BufferUsageFlag::CopyDst => vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsageFlag::CopySrc => vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsageFlag::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferUsageFlag::Index => vk::BufferUsageFlags::INDEX_BUFFER,
            BufferUsageFlag::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferUsageFlag::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }

    pub(crate) fn flags_to_vk(bflags: &BitFlags<Self>) -> vk::BufferUsageFlags {
        let mut out = vk::BufferUsageFlags::empty();
        for fl in bflags.iter() {
            out |= fl.to_vk();
        }
        out
    }
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct BufferCreateInfo {
    pub size: u64,
    #[builder(default=BitFlags::empty())]
    pub used_for: BitFlags<BufferUsageFlag>,
    #[builder(default=DataMoveDir::None)]
    pub data_move_dir: DataMoveDir,
}

pub(crate) struct BufferDropper {
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

#[derive(Clone, Getters)]
pub struct Buffer {
    pub(crate) dropper: Arc<BufferDropper>,
    #[getset(get = "pub")]
    info: BufferCreateInfo,
}

impl Buffer {
    pub(crate) fn new(
        dd: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        info: BufferCreateInfo,
    ) -> anyhow::Result<Self> {
        let usage_flags = BufferUsageFlag::flags_to_vk(&info.used_for);
        let handle = unsafe {
            dd.device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(info.size)
                        .usage(usage_flags),
                    None,
                )
                .context("vk buffer creation failed")?
        };
        let req = unsafe { dd.device.get_buffer_memory_requirements(handle) };
        let mem = GMem::new(allocator, &format!("{handle:?}"), info.data_move_dir, req)?;
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
            }),
            info,
        })
    }

    pub fn len(&self) -> u64 {
        self.info.size
    }

    pub fn write_cpu(&self, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        match self.info.data_move_dir {
            DataMoveDir::Cpu2Gpu => {}
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

    pub fn view(&self, range: Range<u64>) -> BufferView<'_> {
        BufferView {
            buffer: self,
            range,
        }
    }
}

pub struct BufferView<'a> {
    pub buffer: &'a Buffer,
    pub range: Range<u64>,
}

#[bitflags]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ImageUsageFlag {
    CopyDst,
    CopySrc,
    RenderAttach,
    Sampled,
    Storage,
}

impl ImageUsageFlag {
    pub(crate) fn to_vk(&self, is_depth: bool) -> vk::ImageUsageFlags {
        match self {
            ImageUsageFlag::CopyDst => vk::ImageUsageFlags::TRANSFER_DST,
            ImageUsageFlag::CopySrc => vk::ImageUsageFlags::TRANSFER_SRC,
            ImageUsageFlag::RenderAttach => {
                if is_depth {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                }
            }
            ImageUsageFlag::Sampled => vk::ImageUsageFlags::SAMPLED,
            ImageUsageFlag::Storage => vk::ImageUsageFlags::STORAGE,
        }
    }

    pub(crate) fn flags_to_vk(bflags: &BitFlags<Self>, is_depth: bool) -> vk::ImageUsageFlags {
        let mut out = vk::ImageUsageFlags::empty();
        for fl in bflags.iter() {
            out |= fl.to_vk(is_depth);
        }
        out
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Format {
    Rgba8,
    Bgra8,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba10,
    Bgra10,
    Rgba16,
    Rgba16Float,
    D24S8,
    D32Float,
}

impl Format {
    pub fn is_depth(&self) -> bool {
        match self {
            Self::D24S8 | Self::D32Float => true,
            _ => false,
        }
    }

    pub fn has_stencil(&self) -> bool {
        match self {
            Self::D24S8 => true,
            _ => false,
        }
    }

    pub fn rem_srgb(&self) -> Self {
        match self {
            Self::Rgba8Srgb => Self::Rgba8,
            Self::Bgra8Srgb => Self::Bgra8,
            _ => self.clone(),
        }
    }
}

impl Format {
    pub fn aspect_flag(&self) -> vk::ImageAspectFlags {
        match self {
            Format::D24S8 => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            Format::D32Float => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }

    pub fn to_vk(&self) -> vk::Format {
        match self {
            Format::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            Format::Bgra8 => vk::Format::B8G8R8A8_UNORM,
            Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            Format::Rgba10 => vk::Format::A2R10G10B10_UNORM_PACK32,
            Format::Bgra10 => vk::Format::A2B10G10R10_UNORM_PACK32,
            Format::Rgba16 => vk::Format::R16G16B16A16_UNORM,
            Format::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Format::D24S8 => vk::Format::D24_UNORM_S8_UINT,
            Format::D32Float => vk::Format::D32_SFLOAT,
        }
    }
}

#[derive(Debug, Clone, TypedBuilder)]
pub struct ImageCreateInfo {
    pub format: Format,
    pub res: (u32, u32, u32),
    #[builder(default = 1)]
    pub layers: u32,
    #[builder(default = 1)]
    pub mip_levels: u32,
    #[builder(default=BitFlags::empty())]
    pub used_for: BitFlags<ImageUsageFlag>,
    #[builder(default=DataMoveDir::None)]
    pub data_move_dir: DataMoveDir,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageViewType {
    E2d,
    Cube,
}

impl ImageViewType {
    pub fn to_vk(&self) -> vk::ImageViewType {
        match self {
            ImageViewType::E2d => vk::ImageViewType::TYPE_2D,
            ImageViewType::Cube => vk::ImageViewType::CUBE,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewInfo {
    pub view_type: ImageViewType,
    pub layer_range: Range<u32>,
    pub level_range: Range<u32>,
}

impl ImageViewInfo {
    pub(crate) fn type_vk(&self) -> vk::ImageViewType {
        match self.view_type {
            ImageViewType::E2d => {
                if (self.layer_range.end - self.layer_range.start) == 1 {
                    vk::ImageViewType::TYPE_2D
                } else {
                    vk::ImageViewType::TYPE_2D_ARRAY
                }
            }
            ImageViewType::Cube => vk::ImageViewType::CUBE,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageAccess {
    pub(crate) layout: vk::ImageLayout,
    pub(crate) access_flags: vk::AccessFlags,
    pub(crate) access_stage: vk::PipelineStageFlags,
}

pub(crate) struct ImageDropper {
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

#[derive(Clone, Getters)]
pub struct Image {
    #[getset(get = "pub")]
    pub(crate) info: ImageCreateInfo,
    pub(crate) dropper: Arc<ImageDropper>,
}

impl Image {
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
        let image_usage = ImageUsageFlag::flags_to_vk(&info.used_for, info.format.is_depth());
        let handle = unsafe {
            dd.device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(image_type)
                        .format(info.format.to_vk())
                        .extent(vk::Extent3D {
                            width: info.res.0,
                            height: info.res.1,
                            depth: info.res.2,
                        })
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .mip_levels(info.mip_levels)
                        .array_layers(info.layers)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(image_usage),
                    None,
                )
                .context("image creation failed")?
        };
        let req = unsafe { dd.device.get_image_memory_requirements(handle) };
        let mem = GMem::new(allocator, &format!("{handle:?}"), info.data_move_dir, req)?;
        unsafe {
            dd.device
                .bind_image_memory(handle, mem.allocation.memory(), mem.allocation.offset())
                .context("binding buffer to memory failed")?
        }
        Ok(Self {
            info,
            dropper: Arc::new(ImageDropper {
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
                        .format(self.info.format.to_vk())
                        .components(vk::ComponentMapping::default())
                        .view_type(key.type_vk())
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(self.info.format.aspect_flag())
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
            image: self.clone(),
            handle,
            info: info.clone(),
        })
    }
}

#[derive(Clone)]
pub struct ImageView {
    pub(crate) image: Image,
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
