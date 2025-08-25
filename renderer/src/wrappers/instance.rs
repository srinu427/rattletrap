use ash::{ext, khr, vk};
use thiserror::Error;

fn get_instance_extensions() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        ext::debug_utils::NAME.as_ptr(),
        khr::get_physical_device_properties2::NAME.as_ptr(),
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
    CreateError(#[from] vk::Result),
}

pub struct Instance {
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Instance {
    pub fn new() -> Result<Self, InstanceError> {
        let entry = unsafe { ash::Entry::load()? };

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

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(Self {
            instance,
            _entry: entry,
        })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
