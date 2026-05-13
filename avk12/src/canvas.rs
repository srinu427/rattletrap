use std::sync::{Arc, RwLock};

use anyhow::Context;
use ash::vk;
use getset::{CopyGetters, Getters};
use hashbrown::HashMap;

use crate::{
    device::DeviceDropper,
    resource::{ImageAccess, ImageCreateInfo, ImageDropper, ImageRef},
    sync::{Fence, FencePool, WaitResult},
};

fn sc_img_usage_flags() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::STORAGE
}

fn get_sc_formats(device: &Arc<DeviceDropper>) -> anyhow::Result<Vec<vk::SurfaceFormatKHR>> {
    unsafe {
        device
            .instance_dropper
            .surface_instance
            .get_physical_device_surface_formats(
                device.gpu_info.handle,
                device.instance_dropper.surface,
            )
            .context("getting surface formats failed")
    }
}

fn get_sc_caps(device: &Arc<DeviceDropper>) -> anyhow::Result<vk::SurfaceCapabilitiesKHR> {
    unsafe {
        device
            .instance_dropper
            .surface_instance
            .get_physical_device_surface_capabilities(
                device.gpu_info.handle,
                device.instance_dropper.surface,
            )
            .context("getting surface capabilities failed")
    }
}

fn get_sc_present_modes(device: &Arc<DeviceDropper>) -> anyhow::Result<Vec<vk::PresentModeKHR>> {
    unsafe {
        device
            .instance_dropper
            .surface_instance
            .get_physical_device_surface_present_modes(
                device.gpu_info.handle,
                device.instance_dropper.surface,
            )
            .context("getting surfface present modes failed")
    }
}

const HDR_FORMATS: [vk::Format; 3] = [
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::A2B10G10R10_UNORM_PACK32,
    vk::Format::A2R10G10B10_UNORM_PACK32,
];

const SDR_FORMATS: [vk::Format; 2] = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];

const COLOR_SPACES: [vk::ColorSpaceKHR; 2] = [
    vk::ColorSpaceKHR::SRGB_NONLINEAR,
    vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
];

fn choose_surface_format(
    surface_formats: &Vec<vk::SurfaceFormatKHR>,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let surface_formats: Vec<_> = surface_formats
        .into_iter()
        .filter(|s| COLOR_SPACES.contains(&s.color_space))
        .collect();
    let surface_format = match HDR_FORMATS.iter().find_map(|format| {
        surface_formats.iter().find_map(|s| {
            if s.format == *format {
                return Some(**s);
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
                        if s.format == *format {
                            return Some(**s);
                        }
                        None
                    })
                })
                .context("no supported surface format")?;
            sf
        }
    };
    Ok(surface_format)
}

#[derive(Getters, CopyGetters)]
pub struct CanvasInfo {
    #[getset(get_copy = "pub")]
    res: (u32, u32),
    present_mode: vk::PresentModeKHR,
    #[getset(get_copy = "pub")]
    surf_format: vk::SurfaceFormatKHR,
    transform: vk::SurfaceTransformFlagsKHR,
}

pub(crate) struct SwapchainDropper {
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) sc_images: Vec<ImageRef>,
    dd: Arc<DeviceDropper>,
}

impl Drop for SwapchainDropper {
    fn drop(&mut self) {
        unsafe {
            self.dd
                .swapchain_device
                .destroy_swapchain(self.handle, None);
        }
    }
}

pub struct PresentableImage {
    idx: u32,
    sc_dropper: Arc<SwapchainDropper>,
}

impl PresentableImage {
    pub fn image(&self) -> &ImageRef {
        &self.sc_dropper.sc_images[self.idx as usize]
    }

    pub fn present(self) -> anyhow::Result<()> {
        self.sc_dropper
            .dd
            .instance_dropper
            .window
            .pre_present_notify();
        unsafe {
            self.sc_dropper
                .dd
                .swapchain_device
                .queue_present(
                    self.sc_dropper.dd.gfx_queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.sc_dropper.handle])
                        .image_indices(&[self.idx]),
                )
                .context("presenting swapchain image failed")?;
        }
        Ok(())
    }
}

pub enum NextImageRes {
    Success(PresentableImage),
    NeedCanvasRefresh,
    Error(String),
}

#[derive(Getters)]
pub struct Canvas {
    #[getset(get = "pub")]
    info: CanvasInfo,
    pub(crate) sc_dropper: Arc<SwapchainDropper>,
    aquire_img_fence: Fence,
    _fence_pool: FencePool,
}

impl Canvas {
    pub fn refresh_res(&mut self) -> anyhow::Result<()> {
        let sc_caps = get_sc_caps(&self.sc_dropper.dd)?;
        let mut surface_resolution = sc_caps.current_extent;
        if surface_resolution.width == u32::MAX || surface_resolution.height == u32::MAX {
            let window_res = self.sc_dropper.dd.instance_dropper.window.inner_size();
            surface_resolution.width = window_res.width;
            surface_resolution.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.sc_dropper.dd.instance_dropper.surface)
            .min_image_count(self.sc_dropper.sc_images.len() as _)
            .image_format(self.info.surf_format.format)
            .image_color_space(self.info.surf_format.color_space)
            .image_extent(surface_resolution)
            .image_array_layers(1)
            .image_usage(sc_img_usage_flags())
            .pre_transform(sc_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.info.present_mode)
            .clipped(true)
            .old_swapchain(self.sc_dropper.handle);
        let swapchain = unsafe {
            self.sc_dropper
                .dd
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .context("swapchain creation failed")?
        };
        let images: Vec<_> = unsafe {
            self.sc_dropper
                .dd
                .swapchain_device
                .get_swapchain_images(swapchain)
                .context("get swapchain images failed")?
                .into_iter()
                .map(|img| ImageRef {
                    dropper: Arc::new(ImageDropper {
                        info: ImageCreateInfo::builder()
                            .res((surface_resolution.width, surface_resolution.height, 1))
                            .format(self.info.surf_format.format)
                            .used_for(sc_img_usage_flags())
                            .build(),
                        handle: img,
                        mem: None,
                        dont_drop: true,
                        dd: self.sc_dropper.dd.clone(),
                        view_cache: RwLock::new(HashMap::new()),
                        last_access: RwLock::new(ImageAccess {
                            layout: vk::ImageLayout::UNDEFINED,
                            access_flags: vk::AccessFlags::empty(),
                            access_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                        }),
                    }),
                })
                .collect()
        };

        // Update new swapchain and images
        self.sc_dropper = Arc::new(SwapchainDropper {
            handle: swapchain,
            sc_images: images,
            dd: self.sc_dropper.dd.clone(),
        });
        self.info.res = (surface_resolution.width, surface_resolution.height);
        self.info.transform = sc_caps.current_transform;

        Ok(())
    }

    pub(crate) fn new(dd: &Arc<DeviceDropper>) -> anyhow::Result<Self> {
        let sc_fmts = get_sc_formats(dd)?;
        let sc_caps = get_sc_caps(dd)?;
        let sc_present_modes = get_sc_present_modes(dd)?;
        let surface_format = choose_surface_format(&sc_fmts)?;
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
            let window_res = dd.instance_dropper.window.inner_size();
            surface_resolution.width = window_res.width;
            surface_resolution.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(dd.instance_dropper.surface)
            .min_image_count(swapchain_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_resolution)
            .image_array_layers(1)
            .image_usage(sc_img_usage_flags())
            .pre_transform(sc_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface_present_mode)
            .clipped(true);
        let swapchain = unsafe {
            dd.swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .context("swapchain creation failed")?
        };
        let images: Vec<_> = unsafe {
            dd.swapchain_device
                .get_swapchain_images(swapchain)
                .context("get swapchain images failed")?
                .into_iter()
                .map(|img| ImageRef {
                    dropper: Arc::new(ImageDropper {
                        info: ImageCreateInfo::builder()
                            .res((surface_resolution.width, surface_resolution.height, 1))
                            .format(surface_format.format)
                            .used_for(sc_img_usage_flags())
                            .build(),
                        handle: img,
                        mem: None,
                        dont_drop: true,
                        dd: dd.clone(),
                        view_cache: RwLock::new(HashMap::new()),
                        last_access: RwLock::new(ImageAccess {
                            layout: vk::ImageLayout::UNDEFINED,
                            access_flags: vk::AccessFlags::empty(),
                            access_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                        }),
                    }),
                })
                .collect()
        };
        let fence_pool = FencePool::new(dd);
        let aquire_img_fence = fence_pool.get_fence()?;
        Ok(Self {
            info: CanvasInfo {
                res: (surface_resolution.width, surface_resolution.height),
                present_mode: surface_present_mode,
                surf_format: surface_format,
                transform: sc_caps.current_transform,
            },
            sc_dropper: Arc::new(SwapchainDropper {
                handle: swapchain,
                sc_images: images,
                dd: dd.clone(),
            }),
            _fence_pool: fence_pool,
            aquire_img_fence,
        })
    }

    pub fn get_next_image(&mut self) -> NextImageRes {
        let next_img_res = unsafe {
            self.sc_dropper.dd.swapchain_device.acquire_next_image(
                self.sc_dropper.handle,
                u64::MAX,
                vk::Semaphore::null(),
                self.aquire_img_fence.handle,
            )
        };
        match next_img_res {
            Ok((idx, ref_needed)) => {
                if let WaitResult::Error(e) = self.aquire_img_fence.wait(u64::MAX) {
                    return NextImageRes::Error(format!("wait for fence failed: {e}"));
                }

                if ref_needed {
                    return NextImageRes::NeedCanvasRefresh;
                }
                return NextImageRes::Success(PresentableImage {
                    idx: idx as _,
                    sc_dropper: self.sc_dropper.clone(),
                });
            }
            Err(e) => match e {
                vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    if let WaitResult::Error(e) = self.aquire_img_fence.wait(u64::MAX) {
                        return NextImageRes::Error(format!("wait for fence failed: {e}"));
                    }
                    return NextImageRes::NeedCanvasRefresh;
                }
                _ => {
                    return NextImageRes::Error(format!(
                        "aquiring next swapchain image failed: {e}"
                    ));
                }
            },
        }
    }

    pub fn image_count(&self) -> usize {
        self.sc_dropper.sc_images.len()
    }
}
