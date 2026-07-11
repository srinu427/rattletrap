use std::sync::Arc;

use ash::{khr, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::window::Window;

use crate::{
    swapchain::SwapchainWrap,
    utils::{
        StagingBuffer, allocate_command_buffers, create_command_pool, create_device,
        create_instance, create_surface, select_gpu,
    },
};

mod swapchain;
mod texture;
mod utils;

pub struct RenderingManager {
    deferred_preserve_buffers: Vec<StagingBuffer>,
    deferred_command_buffer: Option<vk::CommandBuffer>,
    reuseable_command_buffers: Vec<vk::CommandBuffer>,
    command_pool: vk::CommandPool,
    swapchain: SwapchainWrap,
    swapchain_device: khr::swapchain::Device,
    graphics_queue: vk::Queue,
    graphics_qf: u32,
    allocator: Allocator,
    device: ash::Device,
    gpu: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_instance: khr::surface::Instance,
    instance: ash::Instance,
    _entry: ash::Entry,
    window: Arc<Window>,
}

impl RenderingManager {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_instance(&entry)?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, window)?;
        let selected_gpu = select_gpu(&instance, &surface_instance, surface)?;
        let (device, graphics_queue) = create_device(&instance, &selected_gpu)?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: selected_gpu.gpu,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        let swapchain_device = khr::swapchain::Device::new(&instance, &device);
        let mut swapchain = SwapchainWrap::new_uninit();
        swapchain.refresh(
            &device,
            selected_gpu.gpu,
            &swapchain_device,
            &surface_instance,
            surface,
            window,
        )?;
        let command_pool = create_command_pool(&device, selected_gpu.graphics_qf)?;
        Ok(Self {
            deferred_preserve_buffers: vec![],
            deferred_command_buffer: None,
            reuseable_command_buffers: Default::default(),
            command_pool,
            swapchain,
            swapchain_device,
            graphics_queue,
            graphics_qf: selected_gpu.graphics_qf,
            allocator,
            device,
            gpu: selected_gpu.gpu,
            surface,
            surface_instance,
            instance,
            _entry: entry,
            window: window.clone(),
        })
    }

    fn get_command_buffers(&mut self, count: usize) -> anyhow::Result<Vec<vk::CommandBuffer>> {
        let reuse_len = self.reuseable_command_buffers.len();
        if count <= reuse_len {
            let cbs = self
                .reuseable_command_buffers
                .drain(count - reuse_len..)
                .collect();
            Ok(cbs)
        } else {
            let new_cbs =
                allocate_command_buffers(&self.device, self.command_pool, count - reuse_len)?;
            let mut cached_cbs: Vec<_> = self.reuseable_command_buffers.drain(..).collect();
            cached_cbs.extend(new_cbs);
            Ok(cached_cbs)
        }
    }

    fn get_deferred_cb(&mut self) -> anyhow::Result<vk::CommandBuffer> {
        let existing_dcb = self.deferred_command_buffer.clone();
        match existing_dcb {
            Some(t) => Ok(t),
            None => {
                let cmd_buffer = self.get_command_buffers(1)?[0];
                unsafe {
                    self.device
                        .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())?;
                }
                self.deferred_command_buffer = Some(cmd_buffer);
                Ok(cmd_buffer)
            }
        }
    }
}

impl Drop for RenderingManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .queue_wait_idle(self.graphics_queue)
                .inspect_err(|e| log::warn!("waiting for graphics queue to be idle failed: {e}"))
                .ok();
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain.destroy(&self.device, &self.swapchain_device);
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
