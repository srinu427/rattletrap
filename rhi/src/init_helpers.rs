use std::ffi::CString;

use ash::{ext, khr, vk};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::RhiError;

fn get_instance_layers() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        c"VK_LAYER_KHRONOS_validation".as_ptr(),
    ]
}

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

pub fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, RhiError> {
    let layers = get_instance_layers();
    let extensions = get_instance_extensions();
    let app_info = vk::ApplicationInfo::default()
        .api_version(vk::API_VERSION_1_2)
        .application_name(c"rattleapp")
        .application_version(0)
        .engine_name(c"rattletrap")
        .engine_version(0);
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
            .map_err(RhiError::CreateInstanceError)?
    };
    Ok(instance)
}

pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR, RhiError> {
    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            window
                .display_handle()
                .map_err(RhiError::WindowHandleError)?
                .as_raw(),
            window
                .window_handle()
                .map_err(RhiError::WindowHandleError)?
                .as_raw(),
            None,
        )
        .map_err(RhiError::CreateSurfaceError)?
    };
    Ok(surface)
}

pub fn safe_str_to_cstring(str: String) -> CString {
    let mut msg = str.into_bytes();
    let cstr = loop {
        match CString::new(msg) {
            Ok(cstr) => break cstr,
            Err(err) => {
                let idx = err.nul_position();
                msg = err.into_vec();
                msg.remove(idx);
            }
        }
    };
    cstr
}
