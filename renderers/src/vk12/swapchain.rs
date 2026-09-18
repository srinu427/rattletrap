use ash::vk::{self, Handle};
use hashbrown::{HashMap, HashSet};

use crate::vk12::{device::GpuCtx, resource::GpuImage};

const COLOR_SPACE_PREF: &[vk::ColorSpaceKHR] = &[
    vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT,
    vk::ColorSpaceKHR::HDR10_ST2084_EXT,
    vk::ColorSpaceKHR::SRGB_NONLINEAR,
];

const FORMAT_PREF: &[vk::Format] = &[
    vk::Format::A2R10G10B10_UNORM_PACK32,
    vk::Format::A2B10G10R10_UNORM_PACK32,
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16B16A16_UNORM,
    vk::Format::R8G8B8A8_SRGB,
    vk::Format::B8G8R8A8_SRGB,
];

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let mut formats_per_cs = HashMap::new();
    for format in formats {
        formats_per_cs
            .entry(format.color_space)
            .or_insert(HashSet::new())
            .insert(format.format);
    }
    for cs in COLOR_SPACE_PREF {
        if let Some(formats) = formats_per_cs.get(cs) {
            for format in FORMAT_PREF {
                if formats.contains(format) {
                    return Ok(vk::SurfaceFormatKHR {
                        format: *format,
                        color_space: *cs,
                    });
                }
            }
        }
    }
    anyhow::bail!("No Supported surface format")
}

pub struct GpuSwapchain {
    pub handle: vk::SwapchainKHR,
    pub res: (u32, u32),
    pub format: vk::SurfaceFormatKHR,
    pub images: Vec<GpuImage>,
    // semaphores: Vec<vk::Semaphore>,
    pub fence: vk::Fence,
}

impl GpuSwapchain {
    pub fn new(ctx: &mut GpuCtx) -> anyhow::Result<Self> {
        let fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        let mut out = Self {
            handle: vk::SwapchainKHR::null(),
            res: (0, 0),
            format: vk::SurfaceFormatKHR::default(),
            images: Default::default(),
            // semaphores: Default::default(),
            fence,
        };
        out.resize(ctx)?;
        Ok(out)
    }

    pub fn resize(&mut self, ctx: &mut GpuCtx) -> anyhow::Result<()> {
        unsafe {
            ctx.device.device_wait_idle()?;
        }

        let res = ctx.inst.window.inner_size();
        let sc_caps = unsafe {
            ctx.inst
                .surface_instance
                .get_physical_device_surface_capabilities(ctx.gpu_info.gpu, ctx.inst.surface)?
        };
        let sc_fmts = unsafe {
            ctx.inst
                .surface_instance
                .get_physical_device_surface_formats(ctx.gpu_info.gpu, ctx.inst.surface)?
        };
        let sc_pms = unsafe {
            ctx.inst
                .surface_instance
                .get_physical_device_surface_present_modes(ctx.gpu_info.gpu, ctx.inst.surface)?
        };
        let format = choose_surface_format(&sc_fmts)?;
        let image_count = if sc_caps.max_image_count == 0 {
            sc_caps.min_image_count + 1
        } else {
            sc_caps.max_image_count.min(sc_caps.min_image_count + 1)
        };
        let present_mode = if sc_pms.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        unsafe {
            ctx.device.device_wait_idle()?;
        }
        for mut image in self.images.drain(..) {
            image.destroy(ctx);
        }
        self.images.clear();
        let swapchain = unsafe {
            ctx.swapchain_device.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .image_array_layers(1)
                    .image_color_space(format.color_space)
                    .image_extent(vk::Extent2D {
                        width: res.width,
                        height: res.height,
                    })
                    .image_format(format.format)
                    .image_usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::STORAGE,
                    )
                    .min_image_count(image_count)
                    .old_swapchain(self.handle)
                    .pre_transform(sc_caps.current_transform)
                    .present_mode(present_mode)
                    .surface(ctx.inst.surface),
                None,
            )?
        };
        if !self.handle.is_null() {
            unsafe {
                ctx.swapchain_device.destroy_swapchain(self.handle, None);
            }
        }
        let images = unsafe { ctx.swapchain_device.get_swapchain_images(swapchain)? };
        // if images.len() > self.semaphores.len() {
        //     let rem_count = images.len() - self.semaphores.len();
        //     for _ in 0..rem_count {
        //         let sem = unsafe {
        //             ctx.device
        //                 .device
        //                 .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        //         };
        //         self.semaphores.push(sem);
        //     }
        // }
        self.handle = swapchain;
        self.images = images
            .into_iter()
            .map(|i| GpuImage {
                handle: i,
                type_: vk::ImageType::TYPE_2D,
                format: format.format,
                res: (res.width, res.height, 1),
                levels: 1,
                allocation: None,
                views: Default::default(),
                access: Default::default(),
            })
            .collect();
        self.res = (res.width, res.height);
        self.format = format;
        Ok(())
    }

    pub fn acquire(&mut self, ctx: &mut GpuCtx) -> anyhow::Result<Option<u32>> {
        let acq_res = unsafe {
            ctx.swapchain_device.acquire_next_image(
                self.handle,
                u64::MAX,
                vk::Semaphore::null(),
                self.fence,
            )
        };
        match acq_res {
            Ok((idx, refresh_needed)) => {
                unsafe {
                    ctx.device.wait_for_fences(&[self.fence], false, u64::MAX)?;
                    ctx.device.reset_fences(&[self.fence])?;
                }
                if refresh_needed {
                    Ok(None)
                } else {
                    Ok(Some(idx))
                }
            }
            Err(e) => match e {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => Ok(None),
                _ => Err(anyhow::Error::new(e)),
            },
        }
    }

    pub fn present(&mut self, ctx: &mut GpuCtx, idx: u32) -> anyhow::Result<()> {
        unsafe {
            ctx.swapchain_device.queue_present(
                ctx.graphics_q.handle,
                &vk::PresentInfoKHR::default()
                    .image_indices(&[idx])
                    .swapchains(&[self.handle]),
            )?;
        }
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        for mut image in self.images.drain(..) {
            image.destroy(ctx);
        }
        unsafe {
            ctx.swapchain_device.destroy_swapchain(self.handle, None);
            ctx.device.destroy_fence(self.fence, None);
        }
    }
}
