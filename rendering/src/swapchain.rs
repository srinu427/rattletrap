use anyhow::{Context, bail};
use ash::{
    khr,
    vk::{self, Handle},
};
use winit::window::Window;

pub struct SwapchainWrap {
    pub res: (u32, u32),
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub image_views: Vec<vk::ImageView>,
    pub images: Vec<vk::Image>,
    pub swapchain: vk::SwapchainKHR,
    pub fence: vk::Fence,
}

impl SwapchainWrap {
    pub fn new_uninit() -> Self {
        Self {
            res: (0, 0),
            format: vk::Format::UNDEFINED,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            present_mode: vk::PresentModeKHR::FIFO,
            image_views: vec![],
            images: vec![],
            swapchain: vk::SwapchainKHR::null(),
            fence: vk::Fence::null(),
        }
    }

    pub fn refresh(
        &mut self,
        device: &ash::Device,
        gpu: vk::PhysicalDevice,
        swapchain_device: &khr::swapchain::Device,
        surface_instance: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        window: &Window,
    ) -> anyhow::Result<()> {
        let sc_caps =
            unsafe { surface_instance.get_physical_device_surface_capabilities(gpu, surface)? };
        let mut surface_resolution = sc_caps.current_extent;
        if surface_resolution.width == u32::MAX || surface_resolution.height == u32::MAX {
            let window_res = window.inner_size();
            surface_resolution.width = window_res.width;
            surface_resolution.height = window_res.height;
        }
        let swapchain_image_count = std::cmp::min(
            sc_caps.min_image_count + 1,
            if sc_caps.max_image_count == 0 {
                std::u32::MAX
            } else {
                sc_caps.max_image_count
            },
        );
        let sc_fmts =
            unsafe { surface_instance.get_physical_device_surface_formats(gpu, surface)? };
        let sc_fmt = select_surface_format(&sc_fmts)?;
        let present_modes =
            unsafe { surface_instance.get_physical_device_surface_present_modes(gpu, surface)? };
        let present_mode = select_present_mode(&present_modes);
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .image_array_layers(1)
            .image_color_space(sc_fmt.color_space)
            .image_extent(surface_resolution)
            .image_format(sc_fmt.format)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .min_image_count(swapchain_image_count)
            .old_swapchain(self.swapchain)
            .pre_transform(sc_caps.current_transform)
            .present_mode(present_mode)
            .surface(surface);
        if !self.swapchain.is_null() {
            unsafe {
                swapchain_device.destroy_swapchain(self.swapchain, None);
            }
        }
        for iv in self.image_views.drain(..) {
            unsafe { device.destroy_image_view(iv, None) };
        }
        let swapchain = unsafe { swapchain_device.create_swapchain(&create_info, None)? };
        let images = unsafe { swapchain_device.get_swapchain_images(swapchain)? };
        let image_views: Vec<_> = images
            .iter()
            .map(|i| unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .components(vk::ComponentMapping::default())
                        .format(self.format)
                        .image(*i)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        )
                        .view_type(vk::ImageViewType::TYPE_2D),
                    None,
                )
            })
            .collect::<Result<_, _>>()?;
        self.color_space = sc_fmt.color_space;
        if self.fence.is_null() {
            self.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        }
        self.format = sc_fmt.format;
        self.image_views = image_views;
        self.images = images;
        self.present_mode = present_mode;
        self.res = (surface_resolution.width, surface_resolution.height);
        self.swapchain = swapchain;
        Ok(())
    }

    pub fn get_next_image(
        &self,
        device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
    ) -> anyhow::Result<Option<u32>> {
        if self.swapchain.is_null() {
            bail!("swapchain has not been initialized")
        }
        let acq_res = unsafe {
            swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                self.fence,
            )
        };
        match acq_res {
            Ok((idx, need_refresh)) => {
                unsafe {
                    device.wait_for_fences(&[self.fence], true, u64::MAX)?;
                    device.reset_fences(&[self.fence])?;
                }
                if need_refresh {
                    Ok(None)
                } else {
                    Ok(Some(idx))
                }
            }
            Err(e) => match e {
                vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    unsafe {
                        device.wait_for_fences(&[self.fence], true, u64::MAX)?;
                        device.reset_fences(&[self.fence])?;
                    }
                    Ok(None)
                }
                _ => bail!("acquire next image failed: {e}"),
            },
        }
    }

    pub fn destroy(&self, device: &ash::Device, swapchain_device: &khr::swapchain::Device) {
        if self.swapchain.is_null() {
            unsafe {
                swapchain_device.destroy_swapchain(self.swapchain, None);
            }
        }
        unsafe {
            device.destroy_fence(self.fence, None);
        }
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

fn select_surface_format(
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

fn select_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    let mut found_mailbox = false;
    let mut found_immediate = false;
    for mode in modes {
        if !found_mailbox && *mode == vk::PresentModeKHR::MAILBOX {
            found_mailbox = true;
        } else if !found_immediate && *mode == vk::PresentModeKHR::IMMEDIATE {
            found_immediate = true;
        }
    }
    if found_mailbox {
        vk::PresentModeKHR::MAILBOX
    } else if found_immediate {
        vk::PresentModeKHR::IMMEDIATE
    } else {
        vk::PresentModeKHR::FIFO
    }
}
