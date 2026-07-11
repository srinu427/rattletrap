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

struct GpuClient {
    semaphore_pool: Vec<(vk::Semaphore, u64)>,
    deferred_preserve_buffers: Vec<StagingBuffer>,
    deferred_command_buffer: Option<vk::CommandBuffer>,
    reuseable_command_buffers: Vec<vk::CommandBuffer>,
    command_pool: vk::CommandPool,
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

impl GpuClient {
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
        let command_pool = create_command_pool(&device, selected_gpu.graphics_qf)?;

        Ok(Self {
            semaphore_pool: Default::default(),
            deferred_preserve_buffers: Default::default(),
            deferred_command_buffer: Default::default(),
            reuseable_command_buffers: Default::default(),
            command_pool: command_pool,
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

    fn get_semaphore(&mut self) -> anyhow::Result<(vk::Semaphore, u64)> {
        let existing_sem = self.semaphore_pool.pop();
        match existing_sem {
            Some(t) => Ok(t),
            None => {
                let sem = unsafe {
                    self.device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(
                            &mut vk::SemaphoreTypeCreateInfo::default()
                                .semaphore_type(vk::SemaphoreType::TIMELINE),
                        ),
                        None,
                    )?
                };
                Ok((sem, 0))
            }
        }
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

    fn exec_deferred_cb(&mut self, semaphore: &mut (vk::Semaphore, u64)) -> anyhow::Result<()> {
        if let Some(dcb) = self.deferred_command_buffer.clone() {
            unsafe {
                self.device.end_command_buffer(dcb)?;
                semaphore.1 += 1;
                self.device.queue_submit(
                    self.graphics_queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[dcb])
                        .signal_semaphores(&[semaphore.0])
                        .push_next(
                            &mut vk::TimelineSemaphoreSubmitInfo::default()
                                .signal_semaphore_values(&[semaphore.1]),
                        )],
                    vk::Fence::null(),
                )?;
            }
        }
        Ok(())
    }
    fn reset_deferred_resources(&mut self) -> anyhow::Result<()> {
        for sb in self.deferred_preserve_buffers.drain(..) {
            sb.destroy(&self.device, &mut self.allocator);
        }
        if let Some(dcb) = self.deferred_command_buffer.take() {
            unsafe {
                self.device
                    .reset_command_buffer(dcb, vk::CommandBufferResetFlags::empty())?;
            }
        }
        Ok(())
    }
}

impl Drop for GpuClient {
    fn drop(&mut self) {
        unsafe {
            self.device
                .queue_wait_idle(self.graphics_queue)
                .inspect_err(|e| log::warn!("waiting for graphics queue to be idle failed: {e}"))
                .ok();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct RenderingManager {
    swapchain: SwapchainWrap,
    swapchain_device: khr::swapchain::Device,
    client: GpuClient,
}

impl RenderingManager {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let mut client = GpuClient::new(window)?;
        let swapchain_device = khr::swapchain::Device::new(&client.instance, &client.device);
        let mut swapchain = SwapchainWrap::new_uninit();

        swapchain.refresh(&mut client, &swapchain_device)?;
        Ok(Self {
            swapchain,
            swapchain_device,
            client,
        })
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        self.swapchain
            .refresh(&mut self.client, &self.swapchain_device)
            .inspect_err(|e| log::warn!("refreshing swapchain failed: {e}"))
            .ok();
        Ok(())
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let get_img_res = self
            .swapchain
            .get_next_image(&self.client.device, &self.swapchain_device)?;
        let Some(idx) = get_img_res else {
            self.resize()?;
            return Ok(());
        };
        let mut semaphore = self.client.get_semaphore()?;
        self.client.exec_deferred_cb(&mut semaphore)?;
        unsafe {
            self.client.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[semaphore.0])
                    .values(&[semaphore.1]),
                u64::MAX,
            )?;
        }
        self.client.reset_deferred_resources()?;
        unsafe {
            self.swapchain_device.queue_present(
                self.client.graphics_queue,
                &vk::PresentInfoKHR::default()
                    .swapchains(&[self.swapchain.swapchain])
                    .image_indices(&[idx]),
            )?;
        }
        Ok(())
    }
}

impl Drop for RenderingManager {
    fn drop(&mut self) {
        unsafe {
            self.client
                .device
                .queue_wait_idle(self.client.graphics_queue)
                .inspect_err(|e| log::warn!("waiting for graphics queue to be idle failed: {e}"))
                .ok();
            self.swapchain
                .destroy(&self.client.device, &self.swapchain_device);
        }
    }
}
