use std::{
    sync::{Arc, Mutex},
    u64,
};

use ash::vk;
use rhi2::{
    image::{Format, ImageView as _},
    swapchain::{SCImageRes, SwapchainErr},
};

use crate::{
    command::{CmdBuffer, CmdBufferGen, CommandRecorder},
    device::DeviceDropper,
    image::{Image, ImageAccess, ImageView, rhi2_fmt_to_vk_fmt},
    sync::TaskFuture,
};

fn sc_img_usage_flags() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::STORAGE
}

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
    res: (u32, u32),
    fence: vk::Fence,
    sems: Vec<vk::Semaphore>,
    cmd_pool: Arc<CmdBufferGen>,
    device_dropper: Arc<DeviceDropper>,
}

impl Swapchain {
    fn wrap_img(
        device_dropper: &Arc<DeviceDropper>,
        img: vk::Image,
        format: rhi2::image::Format,
        res: (u32, u32),
    ) -> Image {
        Image {
            handle: img,
            format,
            res: (res.0, res.1, 1),
            layers: 1,
            memory: None,
            flags: rhi2::image::ImageFlags::Storage
                | rhi2::image::ImageFlags::CopyDst
                | rhi2::image::ImageFlags::RenderAttach,
            host_access: rhi2::HostAccess::None,
            device_dropper: device_dropper.clone(),
            last_access: Mutex::new(ImageAccess {
                layout: vk::ImageLayout::UNDEFINED,
                access: vk::AccessFlags::empty(),
                psf: vk::PipelineStageFlags::ALL_COMMANDS,
            }),
        }
    }

    pub fn new(
        device: &Arc<DeviceDropper>,
        cmd_pool: &Arc<CmdBufferGen>,
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
            .image_usage(sc_img_usage_flags())
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
        let res = (surface_resolution.width, surface_resolution.height);
        let images: Vec<_> = unsafe {
            device
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(|e| format!("get swapchain images failed: {e}"))?
                .into_iter()
                .map(|img| Self::wrap_img(device, img, swapchain_format, res))
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
        let fence = unsafe {
            device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("fence creation feailed: {e}"))?
        };
        let mut sems = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            let sem = unsafe {
                device
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| format!("fence creation feailed: {e}"))?
            };
            sems.push(sem);
        }
        Ok(Self {
            handle: swapchain,
            format: surface_format,
            rhi2_fmt: swapchain_format,
            present_mode: surface_present_mode,
            images,
            views,
            res,
            fence,
            sems,
            cmd_pool: cmd_pool.clone(),
            device_dropper: device.clone(),
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
            .image_usage(sc_img_usage_flags())
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
        let res = (surface_resolution.width, surface_resolution.height);
        let new_images: Vec<_> = unsafe {
            self.device_dropper
                .swapchain_device
                .get_swapchain_images(new_sc)
                .map_err(|e| format!("get swapchain images failed: {e}"))?
                .into_iter()
                .map(|img| Self::wrap_img(&self.device_dropper, img, self.rhi2_fmt, res))
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

    fn wait_fence(&self) -> Result<(), String> {
        unsafe {
            self.device_dropper
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("waiting for fence failed: {e}"))?;
            self.device_dropper
                .device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("resetting fence failed: {e}"))?;
        }
        Ok(())
    }
}

impl rhi2::swapchain::Swapchain for Swapchain {
    type I = Image;

    type IV = ImageView;

    type CR = CommandRecorder;

    type SI = SwapchainImage;

    type TF = TaskFuture;

    fn res(&self) -> (u32, u32) {
        self.res
    }

    fn refresh_res(&mut self) -> Result<(), SwapchainErr> {
        self.refresh_res()
            .map_err(|e| format!("refreshing swapchain res failed: {e}"))
            .map_err(rhi2::swapchain::SwapchainErr::ResRefreshErr)
    }

    fn fmt(&self) -> Format {
        self.rhi2_fmt
    }

    fn img_count(&self) -> usize {
        self.images.len()
    }

    fn next_image(&mut self) -> SCImageRes<Self::SI> {
        let acq_res = unsafe {
            self.device_dropper.swapchain_device.acquire_next_image(
                self.handle,
                u64::MAX,
                vk::Semaphore::null(),
                self.fence,
            )
        };
        match acq_res {
            Ok((idx, outdated)) => {
                if !outdated {
                    if let Err(e) = self.wait_fence() {
                        return SCImageRes::Error(e);
                    }
                    return SCImageRes::Success(SwapchainImage {
                        view: self.views[idx as usize].clone(),
                        idx: idx as _,
                        swapchain: self.handle,
                        sem: self.sems[idx as usize],
                        cmd_pool: self.cmd_pool.clone(),
                        device_dropper: self.device_dropper.clone(),
                    });
                } else {
                    return SCImageRes::Outdated;
                }
            }
            Err(e) => match e {
                vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    return SCImageRes::Outdated;
                }
                _ => return SCImageRes::Error(format!("{e}")),
            },
        };
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .swapchain_device
                .destroy_swapchain(self.handle, None);
            self.device_dropper.device.destroy_fence(self.fence, None);
            for sem in self.sems.drain(..) {
                self.device_dropper.device.destroy_semaphore(sem, None);
            }
        }
    }
}

pub struct SwapchainImage {
    pub view: Arc<ImageView>,
    pub idx: usize,
    pub swapchain: vk::SwapchainKHR,
    pub sem: vk::Semaphore,
    pub cmd_pool: Arc<CmdBufferGen>,
    pub device_dropper: Arc<DeviceDropper>,
}

impl rhi2::swapchain::SwapchainImage for SwapchainImage {
    type I = Image;

    type IV = ImageView;

    fn view(&self) -> &Self::IV {
        &self.view
    }

    fn present(&mut self) -> Result<(), SwapchainErr> {
        let mut cmd_buf = CmdBuffer::new_batch(&self.cmd_pool, 1)
            .map(|mut v| v.remove(0))
            .map_err(|e| format!("getting command buffer failed: {e}"))
            .map_err(SwapchainErr::PresentImageErr)?;
        cmd_buf
            .start_recording()
            .map_err(SwapchainErr::PresentImageErr)?;
        let img = self.view.image_holder.as_ref();
        let mut old_access_mut = match img.last_access.lock() {
            Ok(a) => a,
            Err(e) => e.into_inner(),
        };
        let old_access = old_access_mut.clone();
        *old_access_mut = ImageAccess {
            layout: vk::ImageLayout::PRESENT_SRC_KHR,
            access: vk::AccessFlags::MEMORY_READ,
            psf: vk::PipelineStageFlags::ALL_COMMANDS,
        };
        if old_access.layout != vk::ImageLayout::PRESENT_SRC_KHR {
            let aspect_mask = if img.format.is_depth() {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            unsafe {
                self.device_dropper.device.cmd_pipeline_barrier(
                    cmd_buf.handle,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(img.handle)
                        .old_layout(old_access.layout)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_access_mask(old_access.access)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .src_queue_family_index(self.device_dropper.gpu_info.gfx_qf as _)
                        .dst_queue_family_index(self.device_dropper.gpu_info.gfx_qf as _)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect_mask)
                                .level_count(1)
                                .layer_count(img.layers),
                        )],
                );
            }
        }
        cmd_buf
            .stop_recording()
            .map_err(SwapchainErr::PresentImageErr)?;
        unsafe {
            self.device_dropper
                .device
                .queue_submit(
                    self.device_dropper.gfx_queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[cmd_buf.handle])
                        .signal_semaphores(&[self.sem])],
                    vk::Fence::null(),
                )
                .map_err(|e| format!("queue submission failed:{e}"))
                .map_err(SwapchainErr::PresentImageErr)?;
            self.device_dropper
                .swapchain_device
                .queue_present(
                    self.device_dropper.gfx_queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.swapchain])
                        .image_indices(&[self.idx as _])
                        .wait_semaphores(&[self.sem]),
                )
                .map_err(|e| format!("image present failed: {e}"))
                .map_err(SwapchainErr::PresentImageErr)?;
        }
        Ok(())
    }
}
