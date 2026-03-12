use std::sync::Arc;

use ash::{khr, vk};
use winit::window::Window;

use crate::init_helpers;

#[derive(Debug, Clone)]
pub struct VkGpuInfo {
    pub id: usize,
    pub name: String,
    pub dvram: u64,
    pub is_dedicated: bool,
    pub handle: vk::PhysicalDevice,
    pub gfx_qf: usize,
}

pub struct InstanceDropper {
    pub gpus: Vec<VkGpuInfo>,
    pub surface: vk::SurfaceKHR,
    pub surface_instance: khr::surface::Instance,
    pub instance: ash::Instance,
    pub window: Arc<Window>,
    _entry: ash::Entry,
}

impl InstanceDropper {
    fn new(window: &Arc<Window>) -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load().map_err(|e| format!("vulkan load failed: {e}"))? };
        let instance = init_helpers::create_instance(&entry)
            .map_err(|e| format!("create instance failed: {e}"))?;
        let surface = init_helpers::create_surface(&entry, &instance, &window)
            .map_err(|e| format!("create surface failed: {e}"))?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let gpus = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| format!("getting gpu list failed: {e}"))?
        };
        // Select GPUs supported by surface
        let mut gpu_dets = vec![];
        for gpu in gpus.into_iter() {
            let gpu_props = unsafe { instance.get_physical_device_properties(gpu) };
            let gpu_mem_props = unsafe { instance.get_physical_device_memory_properties(gpu) };
            let qf_props = unsafe { instance.get_physical_device_queue_family_properties(gpu) };
            let is_dedicated = gpu_props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            let gpu_name = gpu_props
                .device_name_as_c_str()
                .map(|cs| cs.to_string_lossy().to_string())
                .unwrap_or(format!("Unknown Device #{}", gpu_dets.len()));
            let dvram: u64 = gpu_mem_props
                .memory_heaps
                .iter()
                .filter(|mh| mh.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|mh| mh.size)
                .sum();
            if let Some((gfx_qf_id, _)) = qf_props
                .into_iter()
                .enumerate()
                .filter(|(i, _)| unsafe {
                    surface_instance
                        .get_physical_device_surface_support(gpu, *i as _, surface)
                        .unwrap_or(false)
                })
                .filter(|(_, qfp)| qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .max_by_key(|(_, qfp)| qfp.queue_count)
            {
                gpu_dets.push(VkGpuInfo {
                    id: gpu_dets.len(),
                    name: gpu_name,
                    dvram,
                    is_dedicated,
                    handle: gpu,
                    gfx_qf: gfx_qf_id,
                });
            }
        }
        Ok(Self {
            gpus: gpu_dets,
            surface,
            surface_instance,
            instance,
            window: window.clone(),
            _entry: entry,
        })
    }
}

pub struct Instance {
    dropper: Arc<InstanceDropper>,
    gpus: Vec<rhi2::GpuInfo>,
}

impl rhi2::Instance for Instance {
    type DType;

    fn get_gpus(&self) -> &Vec<rhi2::GpuInfo> {
        &self.gpus
    }

    fn init_device(self, gpu_id: usize) -> Result<Self::DType, rhi2::InstanceErr> {
        todo!()
    }
}
