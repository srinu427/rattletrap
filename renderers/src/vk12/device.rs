use std::mem::ManuallyDrop;
use std::sync::Arc;

use anyhow::Context;
use ash::ext;
use ash::{khr, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::vk12::resource::{GpuBuffer, GpuImage, image_aspect_mask};

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
        ext::swapchain_colorspace::NAME.as_ptr(),
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

fn create_instance(entry: &ash::Entry) -> anyhow::Result<ash::Instance> {
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
            .with_context(|| "instance creation failed")?
    };
    Ok(instance)
}

fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &Window,
) -> anyhow::Result<vk::SurfaceKHR> {
    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            window
                .display_handle()
                .with_context(|| "get display handle failed")?
                .as_raw(),
            window
                .window_handle()
                .with_context(|| "get window handle failed")?
                .as_raw(),
            None,
        )
        .with_context(|| "surface creation failed")?
    };
    Ok(surface)
}

pub struct GpuInst {
    pub surface: vk::SurfaceKHR,
    pub surface_instance: khr::surface::Instance,
    pub instance: ash::Instance,
    _entry: ash::Entry,
    pub window: Arc<Window>,
}

impl GpuInst {
    fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_instance(&entry)?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, window)?;
        Ok(Self {
            surface,
            surface_instance,
            instance,
            _entry: entry,
            window: window.clone(),
        })
    }
}

impl Drop for GpuInst {
    fn drop(&mut self) {
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn select_gpu(
    instance: &ash::Instance,
    surface_instance: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<DeviceInfo> {
    let gpus = unsafe { instance.enumerate_physical_devices()? };
    let mut supported_gpus = vec![];
    for gpu in &gpus {
        let qf_props = unsafe { instance.get_physical_device_queue_family_properties(*gpu) };
        let graphics_qf = qf_props
            .into_iter()
            .enumerate()
            .filter(|(_idx, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .filter(|(idx, _props)| unsafe {
                surface_instance
                    .get_physical_device_surface_support(*gpu, *idx as _, surface)
                    .unwrap_or(false)
            })
            .max_by_key(|(_idx, props)| props.queue_count);
        if let Some((qf, qf_prop)) = graphics_qf {
            let gpu_props = unsafe { instance.get_physical_device_properties(*gpu) };
            let mem_props = unsafe { instance.get_physical_device_memory_properties(*gpu) };
            supported_gpus.push((gpu, qf, qf_prop, gpu_props, mem_props));
        }
    }
    supported_gpus.sort_by_key(|g| {
        g.4.memory_heaps_as_slice()
            .iter()
            .map(|h| h.size)
            .sum::<u64>()
    });
    let selected_gpu = supported_gpus
        .iter()
        .find(|g| g.3.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        .unwrap_or(supported_gpus.first().context("no supported gpus found")?);
    let gpu_name = selected_gpu
        .3
        .device_name_as_c_str()
        .map_or("UNKNOWN_DEVICE_NAME".to_string(), |cs| {
            cs.to_string_lossy().to_string()
        });
    let mut gpu_vram = 0;
    for mh in selected_gpu.4.memory_heaps_as_slice() {
        if mh.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            gpu_vram += mh.size;
        }
    }
    let mut gpu_unified_mem = false;
    for mt in selected_gpu.4.memory_types_as_slice() {
        if mt
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && mt
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            gpu_unified_mem = true;
            break;
        }
    }
    Ok(DeviceInfo {
        gpu: *selected_gpu.0,
        name: gpu_name,
        vram: gpu_vram,
        graphics_qf: selected_gpu.1 as _,
        unified_mem: gpu_unified_mem,
    })
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub gpu: vk::PhysicalDevice,
    pub name: String,
    pub vram: u64,
    pub graphics_qf: u32,
    pub unified_mem: bool,
}

pub struct GpuCommandRecorder {
    pub is_recording: bool,
    pub cb: vk::CommandBuffer,
    pub preserve_buffers: Vec<GpuBuffer>,
}

impl GpuCommandRecorder {
    fn begin(&mut self, ctx: &mut GpuCtx) -> anyhow::Result<()> {
        if !self.is_recording {
            unsafe {
                ctx.device
                    .begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default())?;
            }
            self.is_recording = true;
        }
        Ok(())
    }

    fn end(&mut self, ctx: &mut GpuCtx) -> anyhow::Result<()> {
        if self.is_recording {
            unsafe {
                ctx.device.end_command_buffer(self.cb)?;
            }
            self.is_recording = false;
        }
        Ok(())
    }

    fn copy_b2b_coherent(
        src: &mut GpuBuffer,
        src_offset: u64,
        dst: &mut GpuBuffer,
        dst_offset: u64,
        len: u64,
    ) -> anyhow::Result<()> {
        let src_mem = src
            .allocation
            .as_mut()
            .with_context(|| "no mem allocation for src buffer")?
            .mapped_slice()
            .with_context(|| "cant map src buffer memory")?;
        let src_slice = &src_mem[src_offset as usize..(src_offset + len) as usize];
        let dst_mem = dst
            .allocation
            .as_mut()
            .with_context(|| "no mem allocation for dst buffer")?
            .mapped_slice_mut()
            .with_context(|| "cant map dst buffer memory")?;
        let dst_slice = &mut dst_mem[dst_offset as usize..(dst_offset + len) as usize];
        dst_slice.copy_from_slice(src_slice);
        Ok(())
    }

    pub fn copy_b2b(
        &mut self,
        ctx: &mut GpuCtx,
        src: &mut GpuBuffer,
        src_offset: u64,
        dst: &mut GpuBuffer,
        dst_offset: u64,
        len: u64,
    ) -> anyhow::Result<()> {
        if ctx.gpu_info.unified_mem {
            Self::copy_b2b_coherent(src, src_offset, dst, dst_offset, len)?;
            return Ok(());
        }
        self.begin(ctx)?;
        unsafe {
            ctx.device.cmd_copy_buffer(
                self.cb,
                src.handle,
                dst.handle,
                &[vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size: len,
                }],
            );
            ctx.device.cmd_pipeline_barrier(
                self.cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier::default()
                    .buffer(dst.handle)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_queue_family_index(ctx.graphics_q.family)
                    .offset(0)
                    .size(dst.len)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(ctx.graphics_q.family)],
                &[],
            );
        }
        Ok(())
    }

    pub fn write_to_gpu_buffer(
        &mut self,
        ctx: &mut GpuCtx,
        buffer: &mut GpuBuffer,
        offset: u64,
        data: &[u8],
    ) -> anyhow::Result<()> {
        self.begin(ctx)?;
        let copy_size = (buffer.len - offset).min(data.len() as _);
        if copy_size == 0 {
            return Ok(());
        }
        if ctx.gpu_info.unified_mem {
            buffer
                .allocation
                .as_mut()
                .with_context(|| "buffer has no memory bound to it")?
                .mapped_slice_mut()
                .with_context(|| "cant map buffer's memory")?
                [offset as usize..(offset + copy_size) as usize]
                .copy_from_slice(&data[..copy_size as usize]);
            Ok(())
        } else {
            let mut stage_buffer =
                GpuBuffer::new(ctx, copy_size, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
            stage_buffer
                .allocation
                .as_mut()
                .with_context(|| "stage buffer has no memory bound to it")?
                .mapped_slice_mut()
                .with_context(|| "cant map stage buffer's memory")?[..copy_size as usize]
                .copy_from_slice(&data[..copy_size as usize]);
            self.copy_b2b(ctx, &mut stage_buffer, 0, buffer, offset, copy_size)?;
            self.preserve_buffers.push(stage_buffer);
            Ok(())
        }
    }

    pub fn write_to_texture(
        &mut self,
        ctx: &mut GpuCtx,
        image: &mut GpuImage,
        data: &[u8],
    ) -> anyhow::Result<()> {
        let copy_size = data.len() as u64;
        let mut stage_buffer =
            GpuBuffer::new(ctx, copy_size, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
        stage_buffer
            .allocation
            .as_mut()
            .with_context(|| "stage buffer has no memory bound to it")?
            .mapped_slice_mut()
            .with_context(|| "cant map stage buffer's memory")?[..copy_size as usize]
            .copy_from_slice(&data[..copy_size as usize]);
        unsafe {
            ctx.device.cmd_copy_buffer_to_image(
                self.cb,
                stage_buffer.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: image_aspect_mask(image.format),
                        base_array_layer: 0,
                        layer_count: 1,
                        mip_level: 0,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: image.res.0,
                        height: image.res.1,
                        depth: image.res.2,
                    },
                }],
            );
        }
        self.preserve_buffers.push(stage_buffer);
        Ok(())
    }

    pub fn submit(mut self, ctx: &mut GpuCtx) -> anyhow::Result<GpuRunningTask> {
        let mut tls = ctx.get_tl_semaphore()?;
        if !self.is_recording {
            return Ok(GpuRunningTask { cr: self, sem: tls });
        }
        self.end(ctx)?;
        tls.last_submitted += 1;
        unsafe {
            ctx.device.queue_submit(
                ctx.graphics_q.handle,
                &[vk::SubmitInfo::default()
                    .command_buffers(&[self.cb])
                    .signal_semaphores(&[tls.handle])
                    .push_next(
                        &mut vk::TimelineSemaphoreSubmitInfo::default()
                            .signal_semaphore_values(&[tls.last_submitted]),
                    )],
                vk::Fence::null(),
            )?;
        }
        Ok(GpuRunningTask { cr: self, sem: tls })
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        if let Err(e) = self.end(ctx) {
            log::error!("ending command buffer {:?} failed: {e}", self.cb);
        };
        ctx.graphics_q.cmd_buffers.push(self.cb);
        for mut buffer in self.preserve_buffers.drain(..) {
            buffer.destroy(ctx);
        }
    }
}

pub struct GpuTlSemaphore {
    handle: vk::Semaphore,
    last_submitted: u64,
}

impl GpuTlSemaphore {
    fn destroy(&self, dev: &ash::Device) {
        unsafe {
            dev.destroy_semaphore(self.handle, None);
        }
    }
}

pub struct GpuRunningTask {
    cr: GpuCommandRecorder,
    sem: GpuTlSemaphore,
}

impl GpuRunningTask {
    pub fn wait(mut self, ctx: &mut GpuCtx) -> anyhow::Result<()> {
        unsafe {
            ctx.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[self.sem.handle])
                    .values(&[self.sem.last_submitted]),
                u64::MAX,
            )?;
        }
        ctx.semaphore_pool.push(self.sem);
        ctx.graphics_q.cmd_buffers.push(self.cr.cb);
        for mut buffer in self.cr.preserve_buffers.drain(..) {
            buffer.destroy(ctx);
        }
        Ok(())
    }
}

pub struct GpuQueue {
    pub handle: vk::Queue,
    pub family: u32,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
}

impl GpuQueue {
    fn new(device: &ash::Device, queue: vk::Queue, queue_family: u32) -> anyhow::Result<Self> {
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family),
                None,
            )?
        };
        Ok(Self {
            handle: queue,
            family: queue_family,
            cmd_pool,
            cmd_buffers: Default::default(),
        })
    }

    fn get_cmd_buffer(&mut self, device: &ash::Device) -> anyhow::Result<vk::CommandBuffer> {
        let cb = self.cmd_buffers.pop();
        let cb = match cb {
            Some(t) => t,
            None => {
                let mut cbs = unsafe {
                    device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(16)
                            .command_pool(self.cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )?
                };
                let cb = cbs.remove(0);
                self.cmd_buffers.extend(cbs);
                cb
            }
        };
        Ok(cb)
    }

    fn destroy(&mut self, device: &ash::Device) {
        self.cmd_buffers.clear();
        unsafe {
            device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}

fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        // ext::descriptor_indexing::NAME.as_ptr(),
        // khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

fn create_device(
    instance: &ash::Instance,
    selected_gpu: &DeviceInfo,
) -> anyhow::Result<(ash::Device, vk::Queue)> {
    let queue_priorities = [1.0];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected_gpu.graphics_qf)
        .queue_priorities(&queue_priorities)];
    let device_extensions = get_device_extensions();
    let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
        // .descriptor_indexing(true)
        // .runtime_descriptor_array(true)
        // .shader_sampled_image_array_non_uniform_indexing(true)
        // .descriptor_binding_sampled_image_update_after_bind(true)
        // .descriptor_binding_partially_bound(true)
        // .descriptor_binding_variable_descriptor_count(true)
        .timeline_semaphore(true);
    let device_features = vk::PhysicalDeviceFeatures::default();
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&device_features)
        .push_next(&mut device_12_features);
    let device = unsafe {
        instance
            .create_device(selected_gpu.gpu, &device_create_info, None)
            .context("vk device creation failed")?
    };
    let gfx_queue = unsafe { device.get_device_queue(selected_gpu.graphics_qf, 0) };
    Ok((device, gfx_queue))
}

pub struct GpuCtx {
    pub semaphore_pool: Vec<GpuTlSemaphore>,
    pub allocator: ManuallyDrop<Allocator>,
    pub graphics_q: GpuQueue,
    pub swapchain_device: khr::swapchain::Device,
    pub device: ash::Device,
    pub gpu_info: DeviceInfo,
    pub inst: GpuInst,
}

impl GpuCtx {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let inst = GpuInst::new(window)?;
        let gpu_info = select_gpu(&inst.instance, &inst.surface_instance, inst.surface)?;
        let (device, queue) = create_device(&inst.instance, &gpu_info)?;
        let graphics_q = GpuQueue::new(&device, queue, gpu_info.graphics_qf)?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: inst.instance.clone(),
            device: device.clone(),
            physical_device: gpu_info.gpu,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            semaphore_pool: Default::default(),
            allocator: ManuallyDrop::new(allocator),
            graphics_q,
            swapchain_device: khr::swapchain::Device::new(&inst.instance, &device),
            device,
            gpu_info,
            inst,
        })
    }

    fn get_tl_semaphore(&mut self) -> anyhow::Result<GpuTlSemaphore> {
        let tls = match self.semaphore_pool.pop() {
            Some(t) => t,
            None => {
                let semaphore = unsafe {
                    self.device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(
                            &mut vk::SemaphoreTypeCreateInfo::default()
                                .initial_value(0)
                                .semaphore_type(vk::SemaphoreType::TIMELINE),
                        ),
                        None,
                    )?
                };
                GpuTlSemaphore {
                    handle: semaphore,
                    last_submitted: 0,
                }
            }
        };
        Ok(tls)
    }

    pub fn get_command_recorder(&mut self) -> anyhow::Result<GpuCommandRecorder> {
        let cb = self.graphics_q.get_cmd_buffer(&self.device)?;
        let mut cr = GpuCommandRecorder {
            is_recording: false,
            cb,
            preserve_buffers: vec![],
        };
        cr.begin(self)?;
        Ok(cr)
    }
}

impl Drop for GpuCtx {
    fn drop(&mut self) {
        for sem in self.semaphore_pool.drain(..) {
            sem.destroy(&self.device);
        }
        self.graphics_q.destroy(&self.device);
        unsafe {
            drop(ManuallyDrop::take(&mut self.allocator));
            self.device.destroy_device(None);
        }
    }
}
