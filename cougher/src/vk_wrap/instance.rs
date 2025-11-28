#[cfg(debug_assertions)]
use ash::ext;
use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug, Clone)]
pub struct Gpu {
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) props: vk::PhysicalDeviceProperties,
    pub(crate) mem_props: vk::PhysicalDeviceMemoryProperties,
    pub(crate) g_queue_family: (usize, vk::QueueFamilyProperties),
}

impl Gpu {
    pub fn name(&self) -> String {
        self.props
            .device_name_as_c_str()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or("Unknown Device Name".to_string())
    }

    pub fn vram(&self) -> u64 {
        self.mem_props
            .memory_heaps_as_slice()
            .iter()
            .filter(|x| x.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|x| x.size)
            .sum()
    }

    pub fn is_dedicated(&self) -> bool {
        self.props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InstanceError {
    #[error("Error loading Vulkan: {0}")]
    EntryLoadError(#[from] ash::LoadingError),
    #[error("Error initializing Vulkan Instance: {0}")]
    InstanceInitError(vk::Result),
    #[error("Error getting window's handles: {0}")]
    WindowHandleError(#[from] raw_window_handle::HandleError),
    #[error("Error initializing Vulkan Instance: {0}")]
    SurfaceInitError(vk::Result),
}

pub struct Instance {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) instance: ash::Instance,
    pub(crate) surface_instance: khr::surface::Instance,
    _entry: ash::Entry,
    pub(crate) window: winit::window::Window,
}

impl Instance {
    fn init_instance(entry: &ash::Entry) -> Result<ash::Instance, InstanceError> {
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
                .map_err(InstanceError::InstanceInitError)?
        };
        Ok(instance)
    }

    fn init_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> Result<vk::SurfaceKHR, InstanceError> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .map_err(InstanceError::SurfaceInitError)?
        };
        Ok(surface)
    }

    pub fn new(window: winit::window::Window) -> Result<Self, InstanceError> {
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

    pub fn list_supported_gpus(&self) -> Vec<Gpu> {
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
                Some(Gpu {
                    physical_device: g,
                    props,
                    mem_props,
                    g_queue_family: g_queue_idx,
                })
            })
            .collect()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
