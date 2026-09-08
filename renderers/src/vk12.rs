use std::fs;
use std::marker::PhantomData;
use std::ops::{AddAssign, Range};
use std::sync::Arc;

use anyhow::Context;
use ash::vk::Handle;
use ash::{ext, khr, vk};
use bytemuck::NoUninit;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use hashbrown::{HashMap, HashSet};
use naga::back::spv;
use naga::front::glsl;
use naga::valid;
use ron::de;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::{Camera3d, Material, Mesh, MeshVertex, Scene};

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

struct GpuInst {
    surface: vk::SurfaceKHR,
    surface_instance: khr::surface::Instance,
    instance: ash::Instance,
    entry: ash::Entry,
    window: Arc<Window>,
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
            entry,
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

struct DeviceInfo {
    gpu: vk::PhysicalDevice,
    name: String,
    vram: u64,
    graphics_qf: u32,
    unified_mem: bool,
}

struct GpuTask(u64);

struct GpuQueue {
    handle: vk::Queue,
    family: u32,
    cmd_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    semaphore: vk::Semaphore,
    last_known_complete: u64,
    last_submitted: u64,
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
        let semaphore = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(
                    &mut vk::SemaphoreTypeCreateInfo::default()
                        .initial_value(0)
                        .semaphore_type(vk::SemaphoreType::TIMELINE),
                ),
                None,
            )?
        };
        Ok(Self {
            handle: queue,
            family: queue_family,
            cmd_pool,
            cmd_buffers: Default::default(),
            semaphore,
            last_known_complete: 0,
            last_submitted: 0,
        })
    }

    fn reclaim(&mut self, cmd_buffer: vk::CommandBuffer) {
        self.cmd_buffers.push(cmd_buffer);
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
            device.destroy_semaphore(self.semaphore, None);
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

struct GpuDevice {
    graphics_q: GpuQueue,
    swapchain_device: khr::swapchain::Device,
    device: ash::Device,
    gpu_info: DeviceInfo,
    inst: GpuInst,
}

impl GpuDevice {
    fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let inst = GpuInst::new(window)?;
        let gpu_info = select_gpu(&inst.instance, &inst.surface_instance, inst.surface)?;
        let (device, queue) = create_device(&inst.instance, &gpu_info)?;
        let graphics_q = GpuQueue::new(&device, queue, gpu_info.graphics_qf)?;

        Ok(Self {
            graphics_q,
            swapchain_device: khr::swapchain::Device::new(&inst.instance, &device),
            device,
            gpu_info,
            inst,
        })
    }

    fn submit(&mut self, cb: vk::CommandBuffer) -> anyhow::Result<GpuTask> {
        self.graphics_q.last_submitted += 1;
        unsafe {
            self.device.queue_submit(
                self.graphics_q.handle,
                &[vk::SubmitInfo::default()
                    .command_buffers(&[cb])
                    .signal_semaphores(&[self.graphics_q.semaphore])
                    .push_next(
                        &mut vk::TimelineSemaphoreSubmitInfo::default()
                            .signal_semaphore_values(&[self.graphics_q.last_submitted]),
                    )],
                vk::Fence::null(),
            )?;
        }
        Ok(GpuTask(self.graphics_q.last_submitted))
    }

    fn wait_for_id(&mut self, idx: u64) -> anyhow::Result<()> {
        if idx > self.graphics_q.last_known_complete {
            unsafe {
                self.device.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.graphics_q.semaphore])
                        .values(&[idx]),
                    u64::MAX,
                )?;
            }
            self.graphics_q.last_known_complete = idx;
        }
        Ok(())
    }

    fn wait(&mut self, tasks: Vec<GpuTask>) -> anyhow::Result<()> {
        let mut max_id = 0;
        for GpuTask(id) in tasks {
            if max_id < id {
                max_id = id;
            }
        }
        self.wait_for_id(max_id)
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        self.graphics_q.destroy(&self.device);
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

const COLOR_SPACE_PREF: &[vk::ColorSpaceKHR] = &[
    vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT,
    vk::ColorSpaceKHR::HDR10_ST2084_EXT,
    vk::ColorSpaceKHR::SRGB_NONLINEAR,
];

const FORMAT_PREF: &[vk::Format] = &[
    vk::Format::A2R10G10B10_UNORM_PACK32,
    vk::Format::A2B10G10R10_UNORM_PACK32,
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16B16A16_UNORM,
    vk::Format::R8G8B8A8_SRGB,
    vk::Format::B8G8R8A8_SRGB,
];

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let mut formats_per_cs = HashMap::new();
    for format in formats {
        formats_per_cs
            .entry(format.color_space)
            .or_insert(HashSet::new())
            .insert(format.format);
    }
    for cs in COLOR_SPACE_PREF {
        if let Some(formats) = formats_per_cs.get(cs) {
            for format in FORMAT_PREF {
                if formats.contains(format) {
                    return Ok(vk::SurfaceFormatKHR {
                        format: *format,
                        color_space: *cs,
                    });
                }
            }
        }
    }
    anyhow::bail!("No Supported surface format")
}

struct GpuSwapchain {
    handle: vk::SwapchainKHR,
    res: (u32, u32),
    format: vk::SurfaceFormatKHR,
    images: Vec<GpuImage>,
    // semaphores: Vec<vk::Semaphore>,
    fence: vk::Fence,
}

impl GpuSwapchain {
    fn new(ctx: &mut GpuCtx) -> anyhow::Result<Self> {
        let fence = unsafe {
            ctx.device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        let mut out = Self {
            handle: vk::SwapchainKHR::null(),
            res: (0, 0),
            format: vk::SurfaceFormatKHR::default(),
            images: Default::default(),
            // semaphores: Default::default(),
            fence,
        };
        out.resize(ctx)?;
        Ok(out)
    }

    fn resize(&mut self, ctx: &mut GpuCtx) -> anyhow::Result<()> {
        ctx.device
            .wait_for_id(ctx.device.graphics_q.last_submitted)?;

        let res = ctx.device.inst.window.inner_size();
        let sc_caps = unsafe {
            ctx.device
                .inst
                .surface_instance
                .get_physical_device_surface_capabilities(
                    ctx.device.gpu_info.gpu,
                    ctx.device.inst.surface,
                )?
        };
        let sc_fmts = unsafe {
            ctx.device
                .inst
                .surface_instance
                .get_physical_device_surface_formats(
                    ctx.device.gpu_info.gpu,
                    ctx.device.inst.surface,
                )?
        };
        let sc_pms = unsafe {
            ctx.device
                .inst
                .surface_instance
                .get_physical_device_surface_present_modes(
                    ctx.device.gpu_info.gpu,
                    ctx.device.inst.surface,
                )?
        };
        let format = choose_surface_format(&sc_fmts)?;
        let image_count = if sc_caps.max_image_count == 0 {
            sc_caps.min_image_count + 1
        } else {
            sc_caps.max_image_count.min(sc_caps.min_image_count + 1)
        };
        let present_mode = if sc_pms.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        unsafe {
            ctx.device.device.device_wait_idle()?;
        }
        for mut image in self.images.drain(..) {
            image.destroy(ctx);
        }
        self.images.clear();
        let swapchain = unsafe {
            ctx.device.swapchain_device.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .image_array_layers(1)
                    .image_color_space(format.color_space)
                    .image_extent(vk::Extent2D {
                        width: res.width,
                        height: res.height,
                    })
                    .image_format(format.format)
                    .image_usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::STORAGE,
                    )
                    .min_image_count(image_count)
                    .old_swapchain(self.handle)
                    .pre_transform(sc_caps.current_transform)
                    .present_mode(present_mode)
                    .surface(ctx.device.inst.surface),
                None,
            )?
        };
        if !self.handle.is_null() {
            unsafe {
                ctx.device
                    .swapchain_device
                    .destroy_swapchain(self.handle, None);
            }
        }
        let images = unsafe {
            ctx.device
                .swapchain_device
                .get_swapchain_images(swapchain)?
        };
        // if images.len() > self.semaphores.len() {
        //     let rem_count = images.len() - self.semaphores.len();
        //     for _ in 0..rem_count {
        //         let sem = unsafe {
        //             ctx.device
        //                 .device
        //                 .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        //         };
        //         self.semaphores.push(sem);
        //     }
        // }
        self.handle = swapchain;
        self.images = images
            .into_iter()
            .map(|i| GpuImage {
                handle: i,
                type_: GpuImageType::E2d,
                format: format.format,
                res: (res.width, res.height, 1),
                levels: 1,
                allocation: None,
                views: Default::default(),
                access: Default::default(),
            })
            .collect();
        self.res = (res.width, res.height);
        self.format = format;
        Ok(())
    }

    fn acquire(&mut self, ctx: &mut GpuCtx) -> anyhow::Result<Option<u32>> {
        let acq_res = unsafe {
            ctx.device.swapchain_device.acquire_next_image(
                self.handle,
                u64::MAX,
                vk::Semaphore::null(),
                self.fence,
            )
        };
        match acq_res {
            Ok((idx, refresh_needed)) => {
                unsafe {
                    ctx.device
                        .device
                        .wait_for_fences(&[self.fence], false, u64::MAX)?;
                    ctx.device.device.reset_fences(&[self.fence])?;
                }
                if refresh_needed {
                    Ok(None)
                } else {
                    Ok(Some(idx))
                }
            }
            Err(e) => match e {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => Ok(None),
                _ => Err(anyhow::Error::new(e)),
            },
        }
    }

    fn present(&mut self, ctx: &mut GpuCtx, idx: u32) -> anyhow::Result<()> {
        unsafe {
            ctx.device.swapchain_device.queue_present(
                ctx.device.graphics_q.handle,
                &vk::PresentInfoKHR::default()
                    .image_indices(&[idx])
                    .swapchains(&[self.handle]),
            )?;
        }
        Ok(())
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        for mut image in self.images.drain(..) {
            image.destroy(ctx);
        }
        unsafe {
            ctx.device
                .swapchain_device
                .destroy_swapchain(self.handle, None);
            ctx.device.device.destroy_fence(self.fence, None);
        }
    }
}

struct GpuCtx {
    allocator: Allocator,
    device: GpuDevice,
}

impl GpuCtx {
    fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let device = GpuDevice::new(window)?;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: device.inst.instance.clone(),
            device: device.device.clone(),
            physical_device: device.gpu_info.gpu,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        Ok(Self { allocator, device })
    }
}

struct GpuBuffer {
    handle: vk::Buffer,
    allocation: Option<Allocation>,
    len: u64,
}

impl GpuBuffer {
    fn new(
        ctx: &mut GpuCtx,
        len: u64,
        usage: vk::BufferUsageFlags,
        cpu_write: bool,
    ) -> anyhow::Result<Self> {
        let handle = unsafe {
            ctx.device.device.create_buffer(
                &vk::BufferCreateInfo::default().size(len).usage(usage),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.device.get_buffer_memory_requirements(handle) };
        let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
            name: &format!("buffer_{:?}", handle),
            requirements,
            location: if cpu_write {
                MemoryLocation::CpuToGpu
            } else {
                MemoryLocation::GpuOnly
            },
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            ctx.device.device.bind_buffer_memory(
                handle,
                allocation.memory(),
                allocation.offset(),
            )?;
        }
        Ok(Self {
            handle,
            allocation: Some(allocation),
            len,
        })
    }

    fn new_gpu_local_ro(
        ctx: &mut GpuCtx,
        len: u64,
        mut usage: vk::BufferUsageFlags,
    ) -> anyhow::Result<Self> {
        if ctx.device.gpu_info.unified_mem {
            GpuBuffer::new(ctx, len, usage, true)
        } else {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
            GpuBuffer::new(ctx, len, usage, false)
        }
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        if let Some(altn) = self.allocation.take() {
            unsafe {
                ctx.device.device.destroy_buffer(self.handle, None);
            }
            if let Err(e) = ctx.allocator.free(altn) {
                log::warn!("freeing memory of buffer {:?} failed: {e}", self.handle)
            };
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum GpuImageType {
    E2d,
    E3d,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct GpuImageViewInfo {
    type_: vk::ImageViewType,
    layer_range: Range<u32>,
    level_range: Range<u32>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct GpuImageAccess {
    layout: vk::ImageLayout,
    access: vk::AccessFlags,
    stage: vk::PipelineStageFlags,
}

impl Default for GpuImageAccess {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            access: vk::AccessFlags::empty(),
            stage: vk::PipelineStageFlags::TOP_OF_PIPE,
        }
    }
}

fn is_depth_stencil(fmt: vk::Format) -> (bool, bool) {
    match fmt {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            (true, false)
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => (true, true),
        _ => (false, false),
    }
}

fn image_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
    let (is_d, has_s) = is_depth_stencil(fmt);
    if is_d {
        if has_s {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

struct GpuImage {
    handle: vk::Image,
    type_: GpuImageType,
    format: vk::Format,
    res: (u32, u32, u32),
    levels: u32,
    allocation: Option<Allocation>,
    views: HashMap<GpuImageViewInfo, vk::ImageView>,
    access: GpuImageAccess,
}

impl GpuImage {
    fn new(
        ctx: &mut GpuCtx,
        type_: GpuImageType,
        format: vk::Format,
        res: (u32, u32, u32),
        levels: u32,
        usage: vk::ImageUsageFlags,
    ) -> anyhow::Result<Self> {
        let handle = unsafe {
            ctx.device.device.create_image(
                &vk::ImageCreateInfo::default()
                    .array_layers(match type_ {
                        GpuImageType::E2d => res.2,
                        GpuImageType::E3d => 1,
                    })
                    .extent(vk::Extent3D {
                        width: res.0,
                        height: res.1,
                        depth: match type_ {
                            GpuImageType::E2d => 1,
                            GpuImageType::E3d => res.2,
                        },
                    })
                    .format(format)
                    .image_type(match type_ {
                        GpuImageType::E2d => vk::ImageType::TYPE_2D,
                        GpuImageType::E3d => vk::ImageType::TYPE_3D,
                    })
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .mip_levels(levels)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(usage),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.device.get_image_memory_requirements(handle) };
        let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
            name: &format!("image_{:?}", handle),
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            ctx.device.device.bind_image_memory(
                handle,
                allocation.memory(),
                allocation.offset(),
            )?;
        }
        Ok(GpuImage {
            handle,
            type_,
            format,
            res,
            levels,
            allocation: Some(allocation),
            views: Default::default(),
            access: Default::default(),
        })
    }

    fn get_view(&mut self, ctx: &GpuCtx, info: GpuImageViewInfo) -> anyhow::Result<vk::ImageView> {
        let iv = self.views.get(&info).cloned();
        let iv = match iv {
            Some(t) => t,
            None => {
                let iv = unsafe {
                    ctx.device.device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .components(vk::ComponentMapping::default())
                            .format(self.format)
                            .image(self.handle)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(image_aspect_mask(self.format))
                                    .base_array_layer(info.layer_range.start)
                                    .base_mip_level(info.level_range.start)
                                    .layer_count(info.layer_range.end - info.layer_range.start)
                                    .level_count(info.layer_range.end - info.layer_range.start),
                            )
                            .view_type(info.type_),
                        None,
                    )?
                };
                self.views.insert(info, iv);
                iv
            }
        };
        Ok(iv)
    }

    fn transition(&mut self, ctx: &GpuCtx, cb: vk::CommandBuffer, new_access: GpuImageAccess) {
        if self.access == new_access {
            return;
        }
        unsafe {
            ctx.device.device.cmd_pipeline_barrier(
                cb,
                self.access.stage,
                new_access.stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(new_access.access)
                    .dst_queue_family_index(ctx.device.gpu_info.graphics_qf)
                    .image(self.handle)
                    .new_layout(new_access.layout)
                    .old_layout(self.access.layout)
                    .src_access_mask(self.access.access)
                    .src_queue_family_index(ctx.device.gpu_info.graphics_qf)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(image_aspect_mask(self.format))
                            .layer_count(self.res.2)
                            .level_count(self.levels),
                    )],
            );
        }
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            for (_, view) in self.views.drain() {
                ctx.device.device.destroy_image_view(view, None);
            }
        }
        if let Some(altn) = self.allocation.take() {
            unsafe {
                ctx.device.device.destroy_image(self.handle, None);
            }
            if let Err(e) = ctx.allocator.free(altn) {
                log::warn!("freeing memory of buffer {:?} failed: {e}", self.handle)
            };
        }
    }
}

fn get_pool_sizes(
    bindings: &[(vk::DescriptorType, u32)],
    sets: u32,
) -> Vec<vk::DescriptorPoolSize> {
    let mut counts = HashMap::new();
    for (t, count) in bindings {
        counts.entry(*t).or_insert(0).add_assign(*count * sets);
    }
    counts
        .iter()
        .map(|(t, c)| {
            vk::DescriptorPoolSize::default()
                .ty(*t)
                .descriptor_count(*c)
        })
        .collect()
}

struct GpuDsl {
    handle: vk::DescriptorSetLayout,
    bindings: Vec<(vk::DescriptorType, u32)>,
    pools: Vec<vk::DescriptorPool>,
    sets: Vec<vk::DescriptorSet>,
}

impl GpuDsl {
    fn new(ctx: &mut GpuCtx, bindings: Vec<(vk::DescriptorType, u32)>) -> anyhow::Result<Self> {
        let dsl = unsafe {
            ctx.device.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(
                    &bindings
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(i as _)
                                .descriptor_count(b.1)
                                .descriptor_type(b.0)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                        })
                        .collect::<Vec<_>>(),
                ),
                None,
            )?
        };
        Ok(Self {
            handle: dsl,
            bindings,
            pools: Default::default(),
            sets: Default::default(),
        })
    }

    fn get_set(&mut self, ctx: &GpuCtx) -> anyhow::Result<vk::DescriptorSet> {
        let set = self.sets.pop();
        let set = match set {
            Some(t) => t,
            None => {
                let pool = unsafe {
                    ctx.device.device.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&get_pool_sizes(&self.bindings, 16))
                            .max_sets(16),
                        None,
                    )?
                };
                let mut sets = unsafe {
                    ctx.device.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(pool)
                            .set_layouts(&[self.handle; 16]),
                    )?
                };
                let set = sets.remove(0);
                self.pools.push(pool);
                self.sets.extend(sets);
                set
            }
        };
        Ok(set)
    }

    fn reclaim(&mut self, set: vk::DescriptorSet) {
        self.sets.push(set);
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            ctx.device
                .device
                .destroy_descriptor_set_layout(self.handle, None);
            for pool in self.pools.drain(..) {
                ctx.device.device.destroy_descriptor_pool(pool, None);
            }
            self.sets.clear();
        }
    }
}

fn convert_glsl_to_spv(glsl_source: &str, stage: naga::ShaderStage) -> anyhow::Result<Vec<u32>> {
    let options = glsl::Options {
        stage,
        defines: Default::default(),
    };

    let mut frontend = glsl::Frontend::default();
    let module = frontend.parse(&options, glsl_source)?;

    let mut validator =
        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all());
    let module_info = validator.validate(&module)?;

    let writer_flags = spv::WriterFlags::empty();
    let mut writer_options = spv::Options::default();
    writer_options.flags = writer_flags;

    let mut writer = spv::Writer::new(&writer_options)?;
    let mut spv_words = vec![];
    writer.write(&module, &module_info, None, &None, &mut spv_words)?;

    Ok(spv_words)
}

fn load_glsl(
    ctx: &GpuCtx,
    code: &str,
    stage: naga::ShaderStage,
) -> anyhow::Result<vk::ShaderModule> {
    let spv_words = convert_glsl_to_spv(code, stage)?;
    let shader_mod = unsafe {
        ctx.device.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&spv_words),
            None,
        )?
    };
    Ok(shader_mod)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct MeshPipelinePushConstant {
    obj_id: u32,
    material_id: u32,
}

struct RenderPipelineVk12 {
    render_pass: vk::RenderPass,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsls: Vec<GpuDsl>,
    framebuffers: HashMap<Vec<vk::ImageView>, vk::Framebuffer>,
}

impl RenderPipelineVk12 {
    fn new_mesh_pipeline(ctx: &mut GpuCtx, format: vk::Format) -> anyhow::Result<Self> {
        let render_pass = unsafe {
            ctx.device.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[
                        vk::AttachmentDescription::default()
                            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .format(format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE),
                        vk::AttachmentDescription::default()
                            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .format(vk::Format::D32_SFLOAT)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE),
                    ])
                    .subpasses(&[vk::SubpassDescription::default()
                        .color_attachments(&[vk::AttachmentReference::default()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
                        .depth_stencil_attachment(
                            &vk::AttachmentReference::default()
                                .attachment(1)
                                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                        )
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]),
                None,
            )?
        };
        let dsls = vec![
            GpuDsl::new(ctx, vec![(vk::DescriptorType::UNIFORM_BUFFER, 1)])?,
            GpuDsl::new(ctx, vec![(vk::DescriptorType::STORAGE_BUFFER, 1)])?,
            GpuDsl::new(ctx, vec![(vk::DescriptorType::STORAGE_BUFFER, 1)])?,
        ];
        let layout = unsafe {
            ctx.device.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[dsls[0].handle, dsls[1].handle, dsls[2].handle])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .size(size_of::<MeshPipelinePushConstant>() as _)
                        .stage_flags(vk::ShaderStageFlags::ALL)]),
                None,
            )?
        };
        let vert_shader = load_glsl(
            ctx,
            include_str!("vk12_shaders/mesh.vert"),
            naga::ShaderStage::Vertex,
        )?;
        let frag_shader = load_glsl(
            ctx,
            include_str!("vk12_shaders/mesh.frag"),
            naga::ShaderStage::Fragment,
        )?;
        let pipeline = unsafe {
            ctx.device
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::default(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .layout(layout)
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .cull_mode(vk::CullModeFlags::NONE)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0)
                                .polygon_mode(vk::PolygonMode::FILL),
                        )
                        .render_pass(render_pass)
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(vert_shader)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::VERTEX),
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(frag_shader)
                                .name(c"main")
                                .stage(vk::ShaderStageFlags::FRAGMENT),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .location(0)
                                        .offset(0),
                                    vk::VertexInputAttributeDescription::default()
                                        .binding(0)
                                        .format(vk::Format::R32G32B32_SFLOAT)
                                        .location(1)
                                        .offset(12),
                                ])
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription::default()
                                        .binding(0)
                                        .input_rate(vk::VertexInputRate::VERTEX)
                                        .stride(size_of::<MeshVertex>() as _),
                                ]),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe {
            ctx.device.device.destroy_shader_module(vert_shader, None);
            ctx.device.device.destroy_shader_module(frag_shader, None);
        }
        Ok(Self {
            render_pass,
            handle: pipeline,
            layout,
            dsls,
            framebuffers: Default::default(),
        })
    }

    fn get_frame_buffer(
        &mut self,
        ctx: &GpuCtx,
        mut images: Vec<&mut GpuImage>,
    ) -> anyhow::Result<vk::Framebuffer> {
        let mut views = vec![];
        for image in &mut images {
            let view = image.get_view(
                &ctx,
                GpuImageViewInfo {
                    type_: vk::ImageViewType::TYPE_2D,
                    layer_range: 0..1,
                    level_range: 0..1,
                },
            )?;
            views.push(view);
        }
        let res = images[0].res;
        let fb = self.framebuffers.get(&views).cloned();
        let fb = match fb {
            Some(t) => t,
            None => {
                let fb = unsafe {
                    ctx.device.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .attachments(&views)
                            .width(res.0)
                            .height(res.1)
                            .layers(1)
                            .render_pass(self.render_pass),
                        None,
                    )?
                };
                if self.framebuffers.len() > 128 {
                    unsafe {
                        for (_, fb) in self.framebuffers.drain() {
                            ctx.device.device.destroy_framebuffer(fb, None);
                        }
                    }
                }
                self.framebuffers.insert(views, fb);
                fb
            }
        };
        Ok(fb)
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            for (_, fb) in self.framebuffers.drain() {
                ctx.device.device.destroy_framebuffer(fb, None);
            }
            ctx.device.device.destroy_pipeline(self.handle, None);
            ctx.device.device.destroy_pipeline_layout(self.layout, None);
            ctx.device
                .device
                .destroy_render_pass(self.render_pass, None);
        }
        for mut dsl in self.dsls.drain(..) {
            dsl.destroy(ctx);
        }
    }
}

fn write_to_gpu_buffer(
    ctx: &mut GpuCtx,
    cb: vk::CommandBuffer,
    buffer: &mut GpuBuffer,
    offset: u64,
    data: &[u8],
) -> anyhow::Result<Option<GpuBuffer>> {
    let copy_size = (buffer.len - offset).min(data.len() as _);
    if copy_size == 0 {
        return Ok(None);
    }
    if ctx.device.gpu_info.unified_mem {
        buffer
            .allocation
            .as_mut()
            .with_context(|| "buffer has no memory bound to it")?
            .mapped_slice_mut()
            .with_context(|| "cant map buffer's memory")?
            [offset as usize..(offset + copy_size) as usize]
            .copy_from_slice(&data[..copy_size as usize]);
        Ok(None)
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
        unsafe {
            ctx.device.device.cmd_copy_buffer(
                cb,
                stage_buffer.handle,
                buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset,
                    size: copy_size,
                }],
            );
        }
        Ok(Some(stage_buffer))
    }
}

struct GpuVecData<T: NoUninit> {
    buffer: GpuBuffer,
    capacity: u32,
    len: u32,
    _phantom: PhantomData<T>,
}

impl<T: NoUninit> GpuVecData<T> {
    fn new(ctx: &mut GpuCtx) -> anyhow::Result<Self> {
        let buf_size = 128 * size_of::<T>() as u64;
        let buffer =
            GpuBuffer::new_gpu_local_ro(ctx, buf_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        Ok(Self {
            buffer,
            capacity: 128,
            len: 0,
            _phantom: Default::default(),
        })
    }

    fn push(
        &mut self,
        ctx: &mut GpuCtx,
        elem: T,
        cb: vk::CommandBuffer,
    ) -> anyhow::Result<Option<GpuBuffer>> {
        let insert_idx = self.len;
        let write_offset = insert_idx as u64 * size_of::<Material>() as u64;
        self.len += 1;
        write_to_gpu_buffer(
            ctx,
            cb,
            &mut self.buffer,
            write_offset,
            bytemuck::bytes_of(&elem),
        )
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        self.buffer.destroy(ctx);
    }
}

struct GpuMaterialMgr {
    materials: HashMap<String, u32>,
    buffer: GpuVecData<Material>,
    dset: vk::DescriptorSet,
}

impl GpuMaterialMgr {
    fn new(ctx: &mut GpuCtx, dpool: &mut GpuDsl) -> anyhow::Result<Self> {
        let buffer = GpuVecData::new(ctx)?;
        let dset = dpool.get_set(ctx)?;
        unsafe {
            ctx.device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(buffer.buffer.handle)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(dset)],
                &[],
            );
        }
        Ok(Self {
            materials: Default::default(),
            buffer,
            dset,
        })
    }

    fn add_material(
        &mut self,
        ctx: &mut GpuCtx,
        name: &str,
        material: Material,
        cb: vk::CommandBuffer,
    ) -> anyhow::Result<Option<GpuBuffer>> {
        if self.materials.contains_key(name) {
            return Ok(None);
        }
        let insert_idx = self.buffer.len;
        let res = self.buffer.push(ctx, material, cb)?;
        self.materials.insert(name.to_string(), insert_idx);
        Ok(res)
    }

    fn destroy(&mut self, ctx: &mut GpuCtx, dpool: &mut GpuDsl) {
        self.buffer.destroy(ctx);
        dpool.reclaim(self.dset);
    }
}

struct GpuLoadedMesh {
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    draw_count: u32,
}

impl GpuLoadedMesh {
    fn new(
        ctx: &mut GpuCtx,
        cb: vk::CommandBuffer,
        mesh: Mesh,
    ) -> anyhow::Result<(Self, Vec<GpuBuffer>)> {
        let vb_size = (mesh.vertices.len() * size_of::<MeshVertex>()) as u64;
        let ib_size = (mesh.indices.len() * size_of::<u16>()) as u64;
        let mut vbo =
            GpuBuffer::new_gpu_local_ro(ctx, vb_size, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let mut ibo =
            GpuBuffer::new_gpu_local_ro(ctx, ib_size, vk::BufferUsageFlags::INDEX_BUFFER)?;
        let draw_count = mesh.indices.len() as u32;
        let mut stg_buffers = vec![];
        if let Some(buf) =
            write_to_gpu_buffer(ctx, cb, &mut vbo, 0, bytemuck::cast_slice(&mesh.vertices))?
        {
            stg_buffers.push(buf);
        }
        if let Some(buf) =
            write_to_gpu_buffer(ctx, cb, &mut ibo, 0, bytemuck::cast_slice(&mesh.indices))?
        {
            stg_buffers.push(buf);
        }
        Ok((
            Self {
                vbo,
                ibo,
                draw_count,
            },
            stg_buffers,
        ))
    }

    fn destroy(&mut self, ctx: &mut GpuCtx) {
        self.vbo.destroy(ctx);
        self.ibo.destroy(ctx);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct GpuMeshInfo {
    transform: glam::Mat4,
}

struct GpuMeshMgr {
    meshes: HashMap<String, GpuLoadedMesh>,
    info_buffer: GpuBuffer,
    dset: vk::DescriptorSet,
}

impl GpuMeshMgr {
    fn new(ctx: &mut GpuCtx, dpool: &mut GpuDsl) -> anyhow::Result<Self> {
        let info_buffer = GpuBuffer::new_gpu_local_ro(
            ctx,
            128 * size_of::<GpuMeshInfo>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let dset = dpool.get_set(ctx)?;
        unsafe {
            ctx.device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(info_buffer.handle)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(dset)],
                &[],
            );
        }
        Ok(Self {
            meshes: Default::default(),
            info_buffer,
            dset,
        })
    }

    fn add_mesh(
        &mut self,
        ctx: &mut GpuCtx,
        name: &str,
        mesh: Mesh,
        cb: vk::CommandBuffer,
    ) -> anyhow::Result<Vec<GpuBuffer>> {
        let (gpu_mesh, stg_buffers) = GpuLoadedMesh::new(ctx, cb, mesh)?;
        self.meshes.insert(name.to_string(), gpu_mesh);
        Ok(stg_buffers)
    }

    fn write_data(
        &mut self,
        ctx: &mut GpuCtx,
        cb: vk::CommandBuffer,
        data: &[GpuMeshInfo],
    ) -> anyhow::Result<Option<GpuBuffer>> {
        let needed_size = (data.len() * size_of::<GpuMeshInfo>()) as u64;
        if self.info_buffer.len < needed_size {
            let info_buffer = GpuBuffer::new_gpu_local_ro(
                ctx,
                needed_size.next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            self.info_buffer.destroy(ctx);
            self.info_buffer = info_buffer;
            unsafe {
                ctx.device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(self.info_buffer.handle)
                            .range(vk::WHOLE_SIZE)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_set(self.dset)],
                    &[],
                );
            }
        }
        write_to_gpu_buffer(
            ctx,
            cb,
            &mut self.info_buffer,
            0,
            bytemuck::cast_slice(data),
        )
    }

    fn destroy(&mut self, ctx: &mut GpuCtx, dpool: &mut GpuDsl) {
        for (_, mut mesh) in self.meshes.drain() {
            mesh.destroy(ctx);
        }
        self.info_buffer.destroy(ctx);
        dpool.reclaim(self.dset);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, NoUninit)]
struct GpuCamera {
    transform: glam::Mat4,
    eye: glam::Vec4,
}

struct GpuLoadedCamera {
    buffer: GpuBuffer,
    dset: vk::DescriptorSet,
}

impl GpuLoadedCamera {
    fn new(ctx: &mut GpuCtx, dpool: &mut GpuDsl) -> anyhow::Result<Self> {
        let buffer = GpuBuffer::new_gpu_local_ro(
            ctx,
            size_of::<GpuCamera>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        let dset = dpool.get_set(ctx)?;
        unsafe {
            ctx.device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(buffer.handle)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(dset)],
                &[],
            );
        }
        Ok(Self { buffer, dset })
    }

    fn udpate(
        &mut self,
        ctx: &mut GpuCtx,
        cb: vk::CommandBuffer,
        cam: &Camera3d,
    ) -> anyhow::Result<Option<GpuBuffer>> {
        let transform = cam.get_perspective_proj();
        let gpu_cam = GpuCamera {
            transform,
            eye: glam::Vec4::from((cam.eye, 1.0)),
        };
        write_to_gpu_buffer(ctx, cb, &mut self.buffer, 0, bytemuck::bytes_of(&gpu_cam))
    }

    fn destroy(&mut self, ctx: &mut GpuCtx, dpool: &mut GpuDsl) {
        self.buffer.destroy(ctx);
        dpool.reclaim(self.dset);
    }
}

pub struct RendererVk12 {
    deferred_buffers_to_delete: Vec<GpuBuffer>,
    deferred_command_buffer: Option<vk::CommandBuffer>,
    loaded_materials: GpuMaterialMgr,
    loaded_meshes: GpuMeshMgr,
    loaded_camera: GpuLoadedCamera,
    depth_image: GpuImage,
    mesh_pipeline: RenderPipelineVk12,
    swapchain: GpuSwapchain,
    ctx: GpuCtx,
}

impl RendererVk12 {
    pub fn new(window: &Arc<Window>) -> anyhow::Result<Self> {
        let mut ctx = GpuCtx::new(window)?;
        let swapchain = GpuSwapchain::new(&mut ctx)?;
        let mut mesh_pipeline =
            RenderPipelineVk12::new_mesh_pipeline(&mut ctx, swapchain.format.format)?;
        let loaded_materials = GpuMaterialMgr::new(&mut ctx, &mut mesh_pipeline.dsls[2])?;
        let loaded_meshes = GpuMeshMgr::new(&mut ctx, &mut mesh_pipeline.dsls[1])?;
        let loaded_camera = GpuLoadedCamera::new(&mut ctx, &mut mesh_pipeline.dsls[0])?;
        let depth_image = GpuImage::new(
            &mut ctx,
            GpuImageType::E2d,
            vk::Format::D32_SFLOAT,
            (swapchain.res.0, swapchain.res.1, 1),
            1,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        Ok(Self {
            deferred_buffers_to_delete: Default::default(),
            deferred_command_buffer: None,
            loaded_materials,
            loaded_meshes,
            loaded_camera,
            depth_image,
            mesh_pipeline,
            swapchain,
            ctx,
        })
    }

    fn get_deferred_cmd_buffer(&mut self) -> anyhow::Result<vk::CommandBuffer> {
        let cb = self.deferred_command_buffer.clone();
        let cb = match cb {
            Some(t) => t,
            None => {
                let cb = self
                    .ctx
                    .device
                    .graphics_q
                    .get_cmd_buffer(&self.ctx.device.device)?;
                unsafe {
                    self.ctx.device.device.begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )?;
                }
                self.deferred_command_buffer = Some(cb);
                cb
            }
        };
        Ok(cb)
    }

    fn load_material(name: &str) -> anyhow::Result<Material> {
        let mat_bytes = fs::read(name)?;
        let material = ron::de::from_bytes(&mat_bytes)?;
        Ok(material)
    }

    pub fn resize(&mut self) -> anyhow::Result<()> {
        self.swapchain.resize(&mut self.ctx)?;
        Ok(())
    }

    pub fn render(&mut self, scene: &Scene, camera: &Camera3d) -> anyhow::Result<()> {
        let Some(idx) = self.swapchain.acquire(&mut self.ctx)? else {
            self.resize()?;
            return Ok(());
        };

        let cb = self.get_deferred_cmd_buffer()?;

        if let Some(stg_buffer) = self
            .loaded_camera
            .udpate(&mut self.ctx, cb, camera)
            .with_context(|| "camera update")?
        {
            self.deferred_buffers_to_delete.push(stg_buffer);
        }
        let object_datas: Vec<_> = scene
            .drawables
            .iter()
            .map(|d| GpuMeshInfo {
                transform: d.transform,
            })
            .collect();
        self.loaded_meshes
            .write_data(&mut self.ctx, cb, &object_datas)?;

        let mut material_ids = vec![];
        for drawable in &scene.drawables {
            match self.loaded_materials.materials.get(&drawable.material) {
                Some(idx) => {
                    material_ids.push(*idx);
                }
                None => {
                    let material = Self::load_material(&drawable.material).unwrap_or_default();
                    let stg_buffer = self.loaded_materials.add_material(
                        &mut self.ctx,
                        &drawable.material,
                        material,
                        cb,
                    )?;
                    material_ids.push(self.loaded_materials.buffer.len);
                    if let Some(buffer) = stg_buffer {
                        self.deferred_buffers_to_delete.push(buffer);
                    }
                }
            };
            if self.loaded_meshes.meshes.get(&drawable.mesh).is_none() {
                let mesh_bytes = fs::read(&drawable.mesh)?;
                let mesh = ron::de::from_bytes(&mesh_bytes)?;
                self.loaded_meshes
                    .add_mesh(&mut self.ctx, &drawable.mesh, mesh, cb)?;
            }
        }

        self.swapchain.images[idx as usize].transition(
            &mut self.ctx,
            cb,
            GpuImageAccess {
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
        );
        if self.depth_image.res != self.swapchain.images[idx as usize].res {
            let depth_image = GpuImage::new(
                &mut self.ctx,
                GpuImageType::E2d,
                vk::Format::D32_SFLOAT,
                (self.swapchain.res.0, self.swapchain.res.1, 1),
                1,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )?;
            self.depth_image.destroy(&mut self.ctx);
            self.depth_image = depth_image;
        }
        self.depth_image.transition(
            &mut self.ctx,
            cb,
            GpuImageAccess {
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            },
        );

        let framebuffer = self.mesh_pipeline.get_frame_buffer(
            &self.ctx,
            vec![
                &mut self.swapchain.images[idx as usize],
                &mut self.depth_image,
            ],
        )?;
        unsafe {
            self.ctx.device.device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue::default(),
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0),
                        },
                    ])
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                        width: self.swapchain.res.0,
                        height: self.swapchain.res.1,
                    }))
                    .render_pass(self.mesh_pipeline.render_pass),
                vk::SubpassContents::INLINE,
            );
            self.ctx.device.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.mesh_pipeline.handle,
            );
            self.ctx.device.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: self.swapchain.res.1 as _,
                    width: self.swapchain.res.0 as _,
                    height: -(self.swapchain.res.1 as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.ctx.device.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(vk::Extent2D {
                        width: self.swapchain.res.0,
                        height: self.swapchain.res.1,
                    })],
            );

            self.ctx.device.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.mesh_pipeline.layout,
                0,
                &[
                    self.loaded_camera.dset,
                    self.loaded_meshes.dset,
                    self.loaded_materials.dset,
                ],
                &[],
            );

            for (i, drawable) in scene.drawables.iter().enumerate() {
                let Some(gpu_mesh) = self.loaded_meshes.meshes.get(&drawable.mesh) else {
                    continue;
                };
                let pc_data = MeshPipelinePushConstant {
                    obj_id: i as _,
                    material_id: material_ids[i],
                };
                self.ctx
                    .device
                    .device
                    .cmd_bind_vertex_buffers(cb, 0, &[gpu_mesh.vbo.handle], &[0]);
                self.ctx.device.device.cmd_bind_index_buffer(
                    cb,
                    gpu_mesh.ibo.handle,
                    0,
                    vk::IndexType::UINT16,
                );
                self.ctx.device.device.cmd_push_constants(
                    cb,
                    self.mesh_pipeline.layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    bytemuck::bytes_of(&pc_data),
                );
                self.ctx
                    .device
                    .device
                    .cmd_draw_indexed(cb, gpu_mesh.draw_count, 1, 0, 0, 0);
            }

            self.ctx.device.device.cmd_end_render_pass(cb);
        }

        self.swapchain.images[idx as usize].transition(
            &mut self.ctx,
            cb,
            GpuImageAccess {
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                access: vk::AccessFlags::empty(),
                stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            },
        );

        unsafe {
            self.ctx.device.device.end_command_buffer(cb)?;
        }

        let task = self.ctx.device.submit(cb)?;
        self.ctx.device.wait(vec![task])?;
        self.ctx.device.graphics_q.reclaim(cb);
        self.deferred_command_buffer = None;
        for mut buf in self.deferred_buffers_to_delete.drain(..) {
            buf.destroy(&mut self.ctx);
        }
        self.swapchain.present(&mut self.ctx, idx)?;
        Ok(())
    }
}

impl Drop for RendererVk12 {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.ctx.device.device.device_wait_idle() {
                log::warn!("waiting for gpu to be idle failed: {e}")
            };
            self.loaded_materials
                .destroy(&mut self.ctx, &mut self.mesh_pipeline.dsls[2]);
            self.loaded_meshes
                .destroy(&mut self.ctx, &mut self.mesh_pipeline.dsls[1]);
            self.loaded_camera
                .destroy(&mut self.ctx, &mut self.mesh_pipeline.dsls[0]);
            self.mesh_pipeline.destroy(&mut self.ctx);
            self.depth_image.destroy(&mut self.ctx);
            self.swapchain.destroy(&mut self.ctx);
        }
    }
}
