use std::sync::{Arc, Mutex};

use ash::vk::{self, Handle};
use rhi2::enumflags2::BitFlags;

use crate::{device::DeviceDropper, memory::Memory};

pub fn rhi2_fmt_to_vk_fmt(fmt: rhi2::image::Format) -> vk::Format {
    match fmt {
        rhi2::image::Format::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        rhi2::image::Format::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        rhi2::image::Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        rhi2::image::Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        rhi2::image::Format::Rgba10 => vk::Format::A2R10G10B10_UNORM_PACK32,
        rhi2::image::Format::Bgra10 => vk::Format::A2B10G10R10_UNORM_PACK32,
        rhi2::image::Format::Rgba16 => vk::Format::R16G16B16A16_UNORM,
        rhi2::image::Format::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        rhi2::image::Format::D24S8 => vk::Format::D24_UNORM_S8_UINT,
        rhi2::image::Format::D32Float => vk::Format::D32_SFLOAT,
    }
}

fn rhi2_flags_to_vk_img_usage_flags(
    format: rhi2::image::Format,
    flags: &BitFlags<rhi2::image::ImageFlags>,
) -> vk::ImageUsageFlags {
    let mut usage = vk::ImageUsageFlags::empty();
    for flag in flags.iter() {
        match flag {
            rhi2::image::ImageFlags::CopyDst => usage |= vk::ImageUsageFlags::TRANSFER_DST,
            rhi2::image::ImageFlags::CopySrc => usage |= vk::ImageUsageFlags::TRANSFER_SRC,
            rhi2::image::ImageFlags::RenderAttach => {
                usage |= if format.is_depth() {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                }
            }
            rhi2::image::ImageFlags::Sampled => usage |= vk::ImageUsageFlags::SAMPLED,
            rhi2::image::ImageFlags::Storage => usage |= vk::ImageUsageFlags::STORAGE,
        }
    }
    usage
}

#[derive(Debug, Clone)]
pub struct ImageAccess {
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags,
    pub psf: vk::PipelineStageFlags,
}

pub struct Image {
    pub handle: vk::Image,
    pub memory: Option<Memory>,
    pub format: rhi2::image::Format,
    pub res: (u32, u32, u32),
    pub layers: u32,
    pub flags: BitFlags<rhi2::image::ImageFlags>,
    pub host_access: rhi2::HostAccess,
    pub device_dropper: Arc<DeviceDropper>,
    pub last_access: Arc<Mutex<ImageAccess>>,
}

impl Image {
    pub fn new(
        device_dropper: &Arc<DeviceDropper>,
        format: rhi2::image::Format,
        res: (u32, u32, u32),
        layers: u32,
        flags: BitFlags<rhi2::image::ImageFlags>,
        host_access: rhi2::HostAccess,
    ) -> Result<Self, String> {
        let extent = vk::Extent3D::default().width(res.0).height(res.1).width(1);
        let image_type = if extent.depth == 1 {
            vk::ImageType::TYPE_2D
        } else {
            vk::ImageType::TYPE_3D
        };
        let create_info = vk::ImageCreateInfo::default()
            .format(rhi2_fmt_to_vk_fmt(format))
            .image_type(image_type)
            .extent(extent)
            .array_layers(layers)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(rhi2_flags_to_vk_img_usage_flags(format, &flags))
            .mip_levels(1);
        let handle = unsafe {
            device_dropper
                .device
                .create_image(&create_info, None)
                .map_err(|e| format!("create vk image failed: {e}"))?
        };
        let reqs = unsafe { device_dropper.device.get_image_memory_requirements(handle) };
        let memory = Memory::new(
            &device_dropper.allocator,
            reqs,
            &format!("{:x}", handle.as_raw()),
            host_access,
        )
        .map_err(|e| format!("mem allocation failed: {e}"))?;
        unsafe {
            device_dropper
                .device
                .bind_image_memory(handle, memory.handle.memory(), memory.handle.offset())
                .map_err(|e| format!("bind mem to buffer failed: {e}"))?;
        }
        Ok(Self {
            handle,
            memory: Some(memory),
            format,
            res,
            layers,
            flags,
            host_access,
            device_dropper: device_dropper.clone(),
            last_access: Arc::new(Mutex::new(ImageAccess {
                layout: vk::ImageLayout::UNDEFINED,
                access: vk::AccessFlags::empty(),
                psf: vk::PipelineStageFlags::ALL_COMMANDS,
            })),
        })
    }
}

impl rhi2::image::Image for Image {
    fn format(&self) -> rhi2::image::Format {
        self.format
    }

    fn res(&self) -> (u32, u32, u32) {
        self.res
    }

    fn layers(&self) -> u32 {
        self.layers
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            if self.memory.is_some() {
                self.device_dropper.device.destroy_image(self.handle, None);
            }
        }
    }
}

fn rhi2_ivt_to_vk(vt: &rhi2::image::ViewType) -> vk::ImageViewType {
    match vt {
        rhi2::image::ViewType::E2d => vk::ImageViewType::TYPE_2D,
        rhi2::image::ViewType::ECube => vk::ImageViewType::CUBE,
    }
}

pub struct ImageView {
    pub handle: vk::ImageView,
    pub image_holder: rhi2::Capped<Image>,
    pub view_type: rhi2::image::ViewType,
}

impl rhi2::image::ImageView for ImageView {
    type IType = Image;

    fn new(
        image: rhi2::Capped<Self::IType>,
        view_type: rhi2::image::ViewType,
    ) -> Result<Self, rhi2::image::ImageViewErr> {
        let image_ref = image.as_ref();
        let aspect_mask = if image_ref.format.is_depth() {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let create_info = vk::ImageViewCreateInfo::default()
            .view_type(rhi2_ivt_to_vk(&view_type))
            .image(image_ref.handle)
            .format(rhi2_fmt_to_vk_fmt(image_ref.format))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .layer_count(image_ref.layers)
                    .level_count(1),
            );
        let handle = unsafe {
            image_ref
                .device_dropper
                .device
                .create_image_view(&create_info, None)
                .map_err(|e| e.to_string())
                .map_err(rhi2::image::ImageViewErr::CreateError)?
        };
        Ok(Self {
            handle,
            image_holder: image,
            view_type,
        })
    }

    fn view_type(&self) -> rhi2::image::ViewType {
        self.view_type
    }

    fn image(&self) -> &rhi2::Capped<Self::IType> {
        &self.image_holder
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.image_holder
                .as_ref()
                .device_dropper
                .device
                .destroy_image_view(self.handle, None);
        }
    }
}
