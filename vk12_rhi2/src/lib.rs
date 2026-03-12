use std::sync::Arc;

use ash::{ext, khr, vk};
use winit::window::Window;

mod buffer;
mod device;
mod init_helpers;
mod instance;
mod memory;
mod swapchain;

struct DeviceDropper {
    swapchain_device: khr::swapchain::Device,
    device: ash::Device,
    gfx_qf_idx: u32,
    gpu: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_instance: khr::surface::Instance,
    instance: ash::Instance,
    window: Arc<Window>,
    _entry: ash::Entry,
}

impl DeviceDropper {
    fn get_surface_formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>, String> {
        unsafe {
            self.surface_instance
                .get_physical_device_surface_formats(self.gpu, self.surface)
                .map_err(|e| format!("get surface formats failed: {e}"))
        }
    }

    fn get_surface_caps(&self) -> Result<vk::SurfaceCapabilitiesKHR, String> {
        unsafe {
            self.surface_instance
                .get_physical_device_surface_capabilities(self.gpu, self.surface)
                .map_err(|e| format!("get surface caps failed: {e}"))
        }
    }

    fn get_surface_present_modes(&self) -> Result<Vec<vk::PresentModeKHR>, String> {
        unsafe {
            self.surface_instance
                .get_physical_device_surface_present_modes(self.gpu, self.surface)
                .map_err(|e| format!("get surface present modes failed: {e}"))
        }
    }
}
