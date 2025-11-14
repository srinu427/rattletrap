use ash::ext;
use ash::{khr, vk};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug, Clone)]
pub struct Vk12Gpu {
    physical_device: vk::PhysicalDevice,
    props: vk::PhysicalDeviceProperties,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    g_queue_family: (usize, vk::QueueFamilyProperties),
}

impl Vk12Gpu {
    fn name(&self) -> String {
        self.props
            .device_name_as_c_str()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or("Unknown Device Name".to_string())
    }

    fn vram(&self) -> u64 {
        self.mem_props
            .memory_heaps_as_slice()
            .iter()
            .filter(|x| x.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|x| x.size)
            .sum()
    }

    fn is_dedicated(&self) -> bool {
        self.props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Vk12InstanceError {
    #[error("Error loading Vulkan: {0}")]
    EntryLoadError(#[from] ash::LoadingError),
    #[error("Error initializing Vulkan Instance: {0}")]
    InstanceInitError(vk::Result),
    #[error("Error getting window's handles: {0}")]
    WindowHandleError(#[from] raw_window_handle::HandleError),
    #[error("Error initializing Vulkan Instance: {0}")]
    SurfaceInitError(vk::Result),
}

pub struct Vk12Instance {
    // Window and Vulkan Instance specific. Stuff initialized before
    surface: vk::SurfaceKHR,
    instance: ash::Instance,
    surface_instance: khr::surface::Instance,
    _entry: ash::Entry,
    window: winit::window::Window,
}

impl Vk12Instance {
    fn init_instance(entry: &ash::Entry) -> Result<ash::Instance, Vk12InstanceError> {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2)
            .application_name(c"Cougher App")
            .application_version(1)
            .engine_name(c"Cougher Vulkan 1.2")
            .engine_version(1);
        let layers = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation".as_ptr(),
        ];
        let extensions = [
            #[cfg(debug_assertions)]
            ext::debug_utils::NAME.as_ptr(),
            khr::surface::NAME.as_ptr(),
            #[cfg(target_os = "windows")]
            khr::win32_surface::NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            khr::xlib_surface::NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            khr::wayland_surface::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            khr::portability_enumeration::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            ext::metal_surface::NAME.as_ptr(),
            #[cfg(target_os = "android")]
            khr::android_surface::NAME.as_ptr(),
        ];

        #[cfg(target_os = "macos")]
        let create_info = vk::InstanceCreateInfo::default()
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        #[cfg(not(target_os = "macos"))]
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(Vk12InstanceError::InstanceInitError)?
        };
        Ok(instance)
    }

    fn init_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> Result<vk::SurfaceKHR, Vk12InstanceError> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .map_err(Vk12InstanceError::SurfaceInitError)?
        };
        Ok(surface)
    }

    pub fn new(window: winit::window::Window) -> Result<Self, Vk12InstanceError> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = Self::init_instance(&entry)?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);

        let surface = match Self::init_surface(&entry, &instance, &window) {
            Ok(s) => s,
            Err(e) => {
                unsafe {
                    instance.destroy_instance(None);
                }
                return Err(e);
            }
        };
        Ok(Self {
            surface,
            instance,
            surface_instance,
            _entry: entry,
            window,
        })
    }

    pub fn list_supported_gpus(&self) -> Vec<Vk12Gpu> {
        let gpus = unsafe { self.instance.enumerate_physical_devices().unwrap_or(vec![]) };
        gpus.into_iter()
            .filter_map(|g| unsafe {
                let props = self.instance.get_physical_device_properties(g);
                let mem_props = self.instance.get_physical_device_memory_properties(g);
                let g_queue_idx = self
                    .instance
                    .get_physical_device_queue_family_properties(g)
                    .into_iter()
                    .enumerate()
                    .filter(|(_, qfp)| qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .filter(|(qid, _)| {
                        self.surface_instance
                            .get_physical_device_surface_support(g, *qid as _, self.surface)
                            .unwrap_or(false)
                    })
                    .min_by_key(|x| x.1.queue_count)?;
                Some(Vk12Gpu {
                    physical_device: g,
                    props,
                    mem_props,
                    g_queue_family: g_queue_idx,
                })
            })
            .collect()
    }
}

impl Drop for Vk12Instance {
    fn drop(&mut self) {
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Vk12SwapchainError {}

#[derive(Debug, thiserror::Error)]
pub enum Vk12RendererError {
    #[error("Instance related error: {0}")]
    Vk12InstanceError(#[from] Vk12InstanceError),
    #[error("Error creating Vulkan Device: {0}")]
    DeviceCreateError(vk::Result),
    #[error("Error getting surface formats: {0}")]
    GetSurfaceFormatsError(vk::Result),
    #[error("Error getting surface capabilities: {0}")]
    GetSurfaceCapabilitiesError(vk::Result),
    #[error("Error getting present modes: {0}")]
    GetPresentModesError(vk::Result),
    #[error("No supported surface formats found")]
    NoSuitableSurfaceFormat,
    #[error("Error creating Vulkan Swapchain: {0}")]
    SwapchainCreateError(vk::Result),
    #[error("Error getting Vulkan Swapchain Images: {0}")]
    SwapchainGetImagesError(vk::Result),
}

pub struct Vk12Renderer {
    g_queue_fam: u32,
    g_queue: vk::Queue,
    physical_device: vk::PhysicalDevice,
    swapchain_device: khr::swapchain::Device,
    device: ash::Device,
    instance: Vk12Instance,
}

impl Vk12Renderer {
    fn init_swapchain(
        instance: &Vk12Instance,
        swapchain_device: &khr::swapchain::Device,
        gpu: &Vk12Gpu,
        window: &winit::window::Window,
    ) -> Result<(vk::SwapchainKHR, Vec<vk::Image>), Vk12RendererError> {
        let (formats, caps, present_modes) = unsafe {
            let formats: Vec<_> = instance
                .surface_instance
                .get_physical_device_surface_formats(gpu.physical_device, instance.surface)
                .map_err(Vk12RendererError::GetSurfaceFormatsError)?
                .into_iter()
                .filter(|format| {
                    let supported = instance
                        .instance
                        .get_physical_device_format_properties(gpu.physical_device, format.format)
                        .optimal_tiling_features
                        .contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_DST,
                        );
                    supported
                })
                .collect();

            let caps = instance
                .surface_instance
                .get_physical_device_surface_capabilities(gpu.physical_device, instance.surface)
                .map_err(Vk12RendererError::GetSurfaceCapabilitiesError)?;

            let present_modes = instance
                .surface_instance
                .get_physical_device_surface_present_modes(gpu.physical_device, instance.surface)
                .map_err(Vk12RendererError::GetPresentModesError)?;
            (formats, caps, present_modes)
        };

        let format = formats
            .iter()
            .filter(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .filter(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    || format.format == vk::Format::B8G8R8A8_SRGB
                    || format.format == vk::Format::R8G8B8A8_UNORM
                    || format.format == vk::Format::R8G8B8A8_SRGB
            })
            .next()
            .cloned()
            .ok_or(Vk12RendererError::NoSuitableSurfaceFormat)?;

        let mut extent = caps.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            let window_res = window.inner_size();
            extent.width = window_res.width;
            extent.height = window_res.height;
        }

        let present_mode = present_modes
            .iter()
            .filter(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .next()
            .cloned()
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // let present_mode = vk::PresentModeKHR::FIFO;

        let swapchain_image_count = std::cmp::min(
            caps.min_image_count + 1,
            if caps.max_image_count == 0 {
                std::u32::MAX
            } else {
                caps.max_image_count
            },
        );

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(instance.surface)
            .min_image_count(swapchain_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe {
            swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(Vk12RendererError::SwapchainCreateError)?
        };

        let swapchain_images = unsafe {
            match swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(Vk12RendererError::SwapchainGetImagesError)
            {
                Ok(imgs) => imgs,
                Err(e) => {
                    swapchain_device.destroy_swapchain(swapchain, None);
                    return Err(e);
                }
            }
        };
        Ok((swapchain, swapchain_images))
    }

    pub fn new(
        instance: Vk12Instance,
        gpu: Vk12Gpu,
    ) -> Result<Self, (Vk12Instance, Vk12RendererError)> {
        let queue_priorities = [0.0];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gpu.g_queue_family.0 as _)
            .queue_priorities(&queue_priorities)];
        let extensions = [
            khr::swapchain::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            khr::portability_subset::NAME.as_ptr(),
        ];
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions);
        let device = unsafe {
            match instance
                .instance
                .create_device(gpu.physical_device, &device_create_info, None)
            {
                Ok(d) => d,
                Err(e) => return Err((instance, Vk12RendererError::DeviceCreateError(e))),
            }
        };

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.g_queue_family.0 as _),
                None,
            )
        };

        let g_queue = unsafe { device.get_device_queue(gpu.g_queue_family.0 as _, 0) };
        let swapchain_device = khr::swapchain::Device::new(&instance.instance, &device);

        Ok(Self {
            g_queue_fam: gpu.g_queue_family.0 as _,
            g_queue,
            physical_device: gpu.physical_device,
            swapchain_device,
            device,
            instance,
        })
    }
}

impl Drop for Vk12Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
