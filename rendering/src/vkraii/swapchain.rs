use std::sync::Arc;

use anyhow::{Context, bail};
use ash::vk;

use crate::vkraii::{
    device::DeviceDropper,
    resource::{ImageAccess, ImageRaii},
};

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
                .with_context(|| "no supported surface format")?;
            sf
        }
    };
    Ok(surface_format)
}

pub struct SwapchainRaii {
    pub res: (u32, u32),
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub images: Vec<ImageRaii>,
    pub swapchain: vk::SwapchainKHR,
    pub fence: vk::Fence,
    pub device_d: Arc<DeviceDropper>,
}

impl SwapchainRaii {
    pub fn new(device_d: &Arc<DeviceDropper>) -> anyhow::Result<Self> {
        let fence = unsafe {
            device_d
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .with_context(|| "vk fence creation failed")?
        };
        let mut out = Self {
            res: Default::default(),
            format: vk::Format::UNDEFINED,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            images: Default::default(),
            swapchain: vk::SwapchainKHR::null(),
            fence,
            device_d: device_d.clone(),
        };
        out.refresh()?;
        Ok(out)
    }

    pub fn refresh(&mut self) -> anyhow::Result<()> {
        let sc_caps = unsafe {
            self.device_d
                .instance_raii
                .surface_instance
                .get_physical_device_surface_capabilities(
                    self.device_d.gpu,
                    self.device_d.instance_raii.surface,
                )
                .with_context(|| "get surface caps failed")?
        };
        let sc_fmts = unsafe {
            self.device_d
                .instance_raii
                .surface_instance
                .get_physical_device_surface_formats(
                    self.device_d.gpu,
                    self.device_d.instance_raii.surface,
                )
                .with_context(|| "get surface fmts failed")?
        };
        let present_modes = unsafe {
            self.device_d
                .instance_raii
                .surface_instance
                .get_physical_device_surface_present_modes(
                    self.device_d.gpu,
                    self.device_d.instance_raii.surface,
                )
                .with_context(|| "get surface present modes failed")?
        };
        let mut sc_res = sc_caps.current_extent;
        if sc_res.width == u32::MAX || sc_res.height == u32::MAX {
            let window_res = self.device_d.instance_raii.window.inner_size();
            sc_res.width = window_res.width;
            sc_res.height = window_res.height;
        }
        let sc_fmt = choose_surface_format(&sc_fmts)?;
        let sc_pm = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        };
        let sc_img_count = std::cmp::min(
            sc_caps.min_image_count + 1,
            if sc_caps.max_image_count == 0 {
                std::u32::MAX
            } else {
                sc_caps.max_image_count
            },
        );
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device_d.instance_raii.surface)
            .min_image_count(sc_img_count)
            .image_format(sc_fmt.format)
            .image_color_space(sc_fmt.color_space)
            .image_extent(sc_res)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .pre_transform(sc_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(sc_pm)
            .old_swapchain(self.swapchain)
            .clipped(true);
        let new_swapchain = unsafe {
            self.device_d
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .with_context(|| "creating vk swapchain failed")?
        };
        let new_images = unsafe {
            self.device_d
                .swapchain_device
                .get_swapchain_images(new_swapchain)
                .with_context(|| "getting swapchain images failed")?
                .into_iter()
                .map(|i| ImageRaii {
                    image: i,
                    memory: None,
                    res: (sc_res.width, sc_res.height, 1),
                    format: sc_fmt.format,
                    layers: 1,
                    levels: 1,
                    access: ImageAccess {
                        access_flags: vk::AccessFlags::empty(),
                        layout: vk::ImageLayout::UNDEFINED,
                        stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                    },
                    views: Default::default(),
                    device_d: self.device_d.clone(),
                })
                .collect()
        };
        unsafe {
            self.device_d
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
        self.res = (sc_res.width, sc_res.height);
        self.format = sc_fmt.format;
        self.color_space = sc_fmt.color_space;
        self.images = new_images;
        self.swapchain = new_swapchain;
        Ok(())
    }

    pub fn acquire_image(&mut self) -> anyhow::Result<Option<PresentImage<'_>>> {
        let next_img_res = unsafe {
            self.device_d.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                self.fence,
            )
        };
        match next_img_res {
            Ok((idx, refresh_needed)) => {
                unsafe {
                    self.device_d
                        .device
                        .wait_for_fences(&[self.fence], true, u64::MAX)?;
                    self.device_d.device.reset_fences(&[self.fence])?;
                }
                if refresh_needed {
                    Ok(None)
                } else {
                    Ok(Some(PresentImage {
                        swapchain: self,
                        idx,
                    }))
                }
            }
            Err(e) => match e {
                vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    unsafe {
                        self.device_d
                            .device
                            .wait_for_fences(&[self.fence], true, u64::MAX)?;
                        self.device_d.device.reset_fences(&[self.fence])?;
                    }
                    Ok(None)
                }
                _ => bail!("getting swapchain image failed: {e}"),
            },
        }
    }
}

impl Drop for SwapchainRaii {
    fn drop(&mut self) {
        self.images.clear();
        unsafe {
            self.device_d
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
            self.device_d.device.destroy_fence(self.fence, None);
        }
    }
}

pub struct PresentImage<'a> {
    swapchain: &'a mut SwapchainRaii,
    idx: u32,
}

impl<'a> PresentImage<'a> {
    pub fn get_image(&mut self) -> &mut ImageRaii {
        &mut self.swapchain.images[self.idx as usize]
    }
}

impl<'a> Drop for PresentImage<'a> {
    fn drop(&mut self) {
        unsafe {
            self.swapchain
                .device_d
                .swapchain_device
                .queue_present(
                    self.swapchain.device_d.graphics_queue,
                    &vk::PresentInfoKHR::default()
                        .image_indices(&[self.idx])
                        .swapchains(&[self.swapchain.swapchain]),
                )
                .inspect_err(|e| log::warn!("presenting swapchain image failed: {e}"))
                .ok();
        }
    }
}
