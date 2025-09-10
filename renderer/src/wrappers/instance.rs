use std::sync::Arc;

use ash::{ext, khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;
use winit::window::Window;

fn get_instance_extensions() -> Vec<*const i8> {
    vec![
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
    ]
}

fn get_instance_layers() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        c"VK_LAYER_KHRONOS_validation".as_ptr(),
    ]
}

#[derive(Debug, Error)]
pub enum InstanceError {
    #[error("Vulkan loading error: {0}")]
    VkLoadError(#[from] ash::LoadingError),
    #[error("Vulkan instance creation error: {0}")]
    InstanceCreateError(vk::Result),
    #[error("Raw window handle error: {0}")]
    RawWindowHandleError(#[from] raw_window_handle::HandleError),
    #[error("Vulkan surface creation error: {0}")]
    SurfaceCreationError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct Instance {
    #[get_copy = "pub"]
    surface: vk::SurfaceKHR,
    #[get = "pub"]
    surface_instance: khr::surface::Instance,
    #[get = "pub"]
    window: Arc<Window>,
    #[get = "pub"]
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Instance {
    pub fn new(window: Arc<Window>) -> Result<Self, InstanceError> {
        let entry = unsafe { ash::Entry::load()? };

        let info = unsafe {
            entry.enumerate_instance_extension_properties(None)
        };
        println!("{:#?}", info);

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Rattletrap")
            .application_version(0)
            .engine_name(c"Rattletrap Engine")
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);

        let layers = get_instance_layers();
        let extensions = get_instance_extensions();

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

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(InstanceError::InstanceCreateError)?;

        let surface_instance = khr::surface::Instance::new(&entry, &instance);

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .map_err(InstanceError::SurfaceCreationError)
        }?;

        Ok(Self {
            surface,
            surface_instance,
            window,
            instance,
            _entry: entry,
        })
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
