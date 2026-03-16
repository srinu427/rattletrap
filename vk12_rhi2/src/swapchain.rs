use std::sync::{Arc, Mutex};

use ash::vk;
use log::warn;
use rhi2::image::{Format, ImageView as _};

use crate::{
    command::CommandRecorder,
    device::DeviceDropper,
    image::{Image, ImageAccess, ImageView, rhi2_fmt_to_vk_fmt},
    sync::{BinSem, CpuFuture, GpuFuture, SyncPool},
};

fn get_sc_formats(device: &Arc<DeviceDropper>) -> Result<Vec<vk::SurfaceFormatKHR>, String> {
    unsafe {
        device
            .instance_dropper
            .surface_instance
            .get_physical_device_surface_formats(
                device.gpu_info.handle,
                device.instance_dropper.surface,
            )
            .map_err(|e| e.to_string())
    }
}

fn get_sc_caps(device: &Arc<DeviceDropper>) -> Result<vk::SurfaceCapabilitiesKHR, String> {
    unsafe {
        device
            .instance_dropper
            .surface_instance
            .get_physical_device_surface_capabilities(
                device.gpu_info.handle,
                device.instance_dropper.surface,
            )
            .map_err(|e| e.to_string())
    }
}

fn get_sc_present_modes(device: &Arc<DeviceDropper>) -> Result<Vec<vk::PresentModeKHR>, String> {
    unsafe {
        device
            .instance_dropper
            .surface_instance
            .get_physical_device_surface_present_modes(
                device.gpu_info.handle,
                device.instance_dropper.surface,
            )
            .map_err(|e| e.to_string())
    }
}

const HDR_FORMATS: [Format; 3] = [Format::Rgba16Float, Format::Bgra10, Format::Rgba10];

const SDR_FORMATS: [Format; 2] = [Format::Bgra8, Format::Rgba8];

const COLOR_SPACES: [vk::ColorSpaceKHR; 2] = [
    vk::ColorSpaceKHR::SRGB_NONLINEAR,
    vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
];

fn choose_surface_format(
    surface_formats: &Vec<vk::SurfaceFormatKHR>,
) -> Result<(vk::SurfaceFormatKHR, Format), String> {
    let surface_formats: Vec<_> = surface_formats
        .into_iter()
        .filter(|s| COLOR_SPACES.contains(&s.color_space))
        .collect();
    let surface_format = match HDR_FORMATS.iter().find_map(|format| {
        surface_formats.iter().find_map(|s| {
            if s.format == rhi2_fmt_to_vk_fmt(*format) {
                return Some((**s, *format));
            }
            None
        })
    }) {
        Some(sf) => sf,
        None => {
            let sf = SDR_FORMATS
                .iter()
                .find_map(|format| {
                    surface_formats.iter().find_map(|s| {
                        if s.format == rhi2_fmt_to_vk_fmt(*format) {
                            return Some((**s, *format));
                        }
                        None
                    })
                })
                .ok_or("No Supported Surface Format")?;
            sf
        }
    };
    Ok(surface_format)
}

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    rhi2_fmt: rhi2::image::Format,
    present_mode: vk::PresentModeKHR,
    images: Vec<Arc<Image>>,
    views: Vec<Arc<ImageView>>,
    gfuts: Vec<Vec<GpuFuture>>,
    cfuts: Vec<Vec<CpuFuture>>,
    res: (u32, u32),
    sync_pool: Arc<SyncPool>,
    device_dropper: Arc<DeviceDropper>,
}

impl Swapchain {
    pub fn new(
        device: &Arc<DeviceDropper>,
        sync_pool: &Arc<SyncPool>,
        old_swapchain: Option<&Swapchain>,
    ) -> Result<Self, String> {
        let sc_fmts = get_sc_formats(device)?;
        let sc_caps = get_sc_caps(device)?;
        let sc_present_modes = get_sc_present_modes(device)?;
        let (surface_format, swapchain_format) = choose_surface_format(&sc_fmts)?;
        let surface_present_mode = sc_present_modes
            .iter()
            .filter(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .next()
            .cloned()
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let swapchain_image_count = std::cmp::min(
            sc_caps.min_image_count + 1,
            if sc_caps.max_image_count == 0 {
                std::u32::MAX
            } else {
                sc_caps.max_image_count
            },
        );
        let mut surface_resolution = sc_caps.current_extent;
        if surface_resolution.width == u32::MAX || surface_resolution.height == u32::MAX {
            let window_res = device.instance_dropper.window.inner_size();
            surface_resolution.width = window_res.width;
            surface_resolution.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(device.instance_dropper.surface)
            .min_image_count(swapchain_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_resolution)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .pre_transform(sc_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface_present_mode)
            .clipped(true)
            .old_swapchain(
                old_swapchain
                    .map(|s| s.handle)
                    .unwrap_or(vk::SwapchainKHR::null()),
            );
        let swapchain = unsafe {
            device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(|e| format!("create Swapchain failed: {e}"))?
        };
        let images: Vec<_> = unsafe {
            device
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(|e| format!("get swapchain images failed: {e}"))?
                .into_iter()
                .map(|img| Image {
                    handle: img,
                    format: swapchain_format,
                    res: (surface_resolution.width, surface_resolution.height, 1),
                    layers: 1,
                    memory: None,
                    flags: rhi2::image::ImageFlags::Storage
                        | rhi2::image::ImageFlags::CopyDst
                        | rhi2::image::ImageFlags::RenderAttach,
                    host_access: rhi2::HostAccess::None,
                    device_dropper: device.clone(),
                    last_access: Arc::new(Mutex::new(ImageAccess {
                        layout: vk::ImageLayout::UNDEFINED,
                        access: vk::AccessFlags::empty(),
                        psf: vk::PipelineStageFlags::ALL_COMMANDS,
                    })),
                })
                .map(Arc::new)
                .collect()
        };
        let views: Vec<_> = images
            .iter()
            .map(|img| {
                ImageView::new(rhi2::Capped::Arc(img.clone()), rhi2::image::ViewType::E2d)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        let cfuts = (0..images.len()).map(|_| Vec::new()).collect();
        let gfuts = (0..images.len()).map(|_| Vec::new()).collect();
        Ok(Self {
            handle: swapchain,
            format: surface_format,
            rhi2_fmt: swapchain_format,
            present_mode: surface_present_mode,
            images,
            views,
            cfuts,
            gfuts,
            res: (surface_resolution.width, surface_resolution.height),
            device_dropper: device.clone(),
            sync_pool: sync_pool.clone(),
        })
    }

    pub fn refresh_res(&mut self) -> Result<(), String> {
        let sc_caps = get_sc_caps(&self.device_dropper)?;
        let mut surface_resolution = sc_caps.current_extent;
        if surface_resolution.width == u32::MAX || surface_resolution.height == u32::MAX {
            let window_res = self.device_dropper.instance_dropper.window.inner_size();
            surface_resolution.width = window_res.width;
            surface_resolution.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device_dropper.instance_dropper.surface)
            .min_image_count(self.images.len() as _)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(surface_resolution)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .pre_transform(sc_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(self.handle);
        let new_sc = unsafe {
            self.device_dropper
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(|e| format!("new swapchain creation failed: {e}"))?
        };
        let new_images: Vec<_> = unsafe {
            self.device_dropper
                .swapchain_device
                .get_swapchain_images(new_sc)
                .map_err(|e| format!("get swapchain images failed: {e}"))?
                .into_iter()
                .map(|img| Image {
                    handle: img,
                    format: self.rhi2_fmt,
                    res: (surface_resolution.width, surface_resolution.height, 1),
                    layers: 1,
                    memory: None,
                    flags: rhi2::image::ImageFlags::Storage
                        | rhi2::image::ImageFlags::CopyDst
                        | rhi2::image::ImageFlags::RenderAttach,
                    host_access: rhi2::HostAccess::None,
                    device_dropper: self.device_dropper.clone(),
                    last_access: Arc::new(Mutex::new(ImageAccess {
                        layout: vk::ImageLayout::UNDEFINED,
                        access: vk::AccessFlags::empty(),
                        psf: vk::PipelineStageFlags::ALL_COMMANDS,
                    })),
                })
                .map(Arc::new)
                .collect()
        };
        let new_views: Vec<_> = new_images
            .iter()
            .map(|img| {
                ImageView::new(rhi2::Capped::Arc(img.clone()), rhi2::image::ViewType::E2d)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        unsafe {
            self.device_dropper
                .swapchain_device
                .destroy_swapchain(self.handle, None);
        }
        self.handle = new_sc;
        self.images = new_images;
        self.views = new_views;

        Ok(())
    }
}

impl rhi2::swapchain::Swapchain for Swapchain {
    type I = Image;

    type IV = ImageView;

    type CR = CommandRecorder;

    type CF = CpuFuture;

    type GF = GpuFuture;

    fn img_count(&self) -> usize {
        self.images.len()
    }

    fn res(&self) -> (u32, u32) {
        self.res
    }

    fn refresh_res(&mut self) {
        self.refresh_res()
            .inspect_err(|e| warn!("refreshing swapchain res failed: {e}"))
            .ok();
    }

    fn fmt(&self) -> Format {
        self.rhi2_fmt
    }

    fn acquire_image_view(&self) -> Result<(&Self::IV, Self::GF), rhi2::swapchain::SwapchainErr> {
        let bin_sem = BinSem::get(&self.sync_pool, 1)
            .map_err(|e| format!("bin sem getting failed: {e}"))
            .map_err(rhi2::swapchain::SwapchainErr::AcquireImageErr)?
            .remove(0);
        let (idx, _) = unsafe {
            match self.device_dropper.swapchain_device.acquire_next_image(
                self.handle,
                u64::MAX,
                bin_sem.handle,
                vk::Fence::null(),
            ) {
                Ok((idx, ref_needed)) => (idx, ref_needed),
                Err(e) => match e {
                    vk::Result::SUBOPTIMAL_KHR => {
                        todo!()
                    }
                    _ => {
                        return Err(rhi2::swapchain::SwapchainErr::AcquireImageErr(format!(
                            "vk acquire next image call failed: {e}"
                        )));
                    }
                },
            }
        };
        let gfut = GpuFuture::from_bin(bin_sem, vec![]);
        Ok((&self.views[idx as usize], gfut))
    }

    fn present(&self, wait_for: Vec<Self::CR>) -> Result<Self::CF, rhi2::swapchain::SwapchainErr> {
        todo!()
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .swapchain_device
                .destroy_swapchain(self.handle, None);
        }
    }
}
