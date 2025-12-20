use std::{
    mem::ManuallyDrop,
    ops::Range,
    sync::{Arc, Mutex},
};

use ash::{LoadingError, ext, khr, vk};
pub use enumflags2;
use enumflags2::{BitFlags, bitflags};
use getset::{CopyGetters, Getters};
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use hashbrown::HashMap;
use log::{info, warn};
use winit::{raw_window_handle::HandleError, window::Window};

mod init_helpers;

#[derive(Debug, thiserror::Error)]
pub enum RhiError {
    #[error("vulkan loading failed: {0}")]
    VkLoadError(#[from] LoadingError),
    #[error("create instance failed: {0}")]
    CreateInstanceError(vk::Result),
    #[error("getting window handle failed: {0}")]
    WindowHandleError(#[from] HandleError),
    #[error("create surface failed: {0}")]
    CreateSurfaceError(vk::Result),
    #[error("get gpus failed: {0}")]
    GetGpusError(vk::Result),
    #[error("no supported gpus found")]
    NoSupportedGpus,
    #[error("create vulkan device failed: {0}")]
    CreateDeviceError(vk::Result),
    #[error("create command pool failed: {0}")]
    CreateCommandPoolError(vk::Result),
    #[error("create command buffer failed: {0}")]
    CreateCommandBufferError(vk::Result),
    #[error("begin command buffer failed: {0}")]
    BeginCommandBufferError(vk::Result),
    #[error("end command buffer failed: {0}")]
    EndCommandBufferError(vk::Result),
    #[error("create fence failed: {0}")]
    CreateFenceError(vk::Result),
    #[error("create semaphore failed: {0}")]
    CreateSemaphoreError(vk::Result),
    #[error("Unsupported Semaphore type for the operation")]
    UnsupportedSemaphoreType,
    #[error("getting surface formats failed: {0}")]
    GetSurfaceFormatsError(vk::Result),
    #[error("no supported surface formats")]
    NoSupportedSurfaceFormat,
    #[error("getting surface capabilities failed: {0}")]
    GetSurfaceCapsError(vk::Result),
    #[error("getting surface present modes failed: {0}")]
    GetSurfacePresentModesError(vk::Result),
    #[error("create swapchain failed: {0}")]
    CreateSwapchainError(vk::Result),
    #[error("getting swapchain images failed: {0}")]
    GetSwapchainImagesError(vk::Result),
    #[error("acquiring swapchain image failed: {0}")]
    AcquireSwapchainImageError(vk::Result),
    #[error("presenting swapchain image failed: {0}")]
    PresentSwapchainImageError(vk::Result),
    #[error("memory allocation failed: {0}")]
    MemAllocError(#[from] AllocationError),
    #[error("memory is not CPU write-able")]
    MemReadOnly,
    #[error("create buffer failed: {0}")]
    CreateBufferError(vk::Result),
    #[error("buffer memory binding failed: {0}")]
    BufferBindMemError(vk::Result),
    #[error("create image failed: {0}")]
    CreateImageError(vk::Result),
    #[error("create image view failed: {0}")]
    CreateImageViewError(vk::Result),
    #[error("image memory binding failed: {0}")]
    ImageBindMemError(vk::Result),
    #[error("create sampler failed: {0}")]
    CreateSamplerError(vk::Result),
    #[error("cycle of tasks found in queue work")]
    CycleInWorkGraph,
    #[error("submitting command buffer to queue failed: {0}")]
    SubmitCommandBufferError(vk::Result),
    #[error("waiting for semaphore on host failed: {0}")]
    WaitSemaphoreError(vk::Result),
    #[error("waiting for fence on host failed: {0}")]
    WaitFenceError(vk::Result),
    #[error("resetting fence failed: {0}")]
    ResetFenceError(vk::Result),
    #[error("creating descriptor set layout failed: {0}")]
    CreateDslError(vk::Result),
    #[error("creating descriptor pool failed: {0}")]
    CreateDPoolError(vk::Result),
    #[error("creating descriptor set failed: {0}")]
    CreateDSetError(vk::Result),
    #[error("creating renderpass failed: {0}")]
    CreateRenderPassError(vk::Result),
    #[error("creating pipeline layout failed: {0}")]
    CreatePipelineLayoutError(vk::Result),
    #[error("creating shader module failed: {0}")]
    CreateShaderModuleError(vk::Result),
    #[error("creating pipeline failed: {0}")]
    CreatePipelineError(vk::Result),
    #[error("creating framebuffer failed: {0}")]
    CreateFramebufferError(vk::Result),
}

fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        ext::descriptor_indexing::NAME.as_ptr(),
        khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

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
    pub fn new(window: &Arc<Window>) -> Result<Self, RhiError> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = init_helpers::create_instance(&entry)?;
        let surface = init_helpers::create_surface(&entry, &instance, &window)?;
        let surface_instance = khr::surface::Instance::new(&entry, &instance);
        let gpus = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(RhiError::GetGpusError)?
        };
        let mut gpu_dets = vec![];
        for gpu in gpus.into_iter() {
            let qf_props = unsafe { instance.get_physical_device_queue_family_properties(gpu) };
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
                gpu_dets.push((gpu, gfx_qf_id as u32));
            }
        }
        if gpu_dets.is_empty() {
            return Err(RhiError::NoSupportedGpus);
        }
        let mut selected_gpu_idx = 0;
        for (idx, (gpu, _)) in gpu_dets.iter().enumerate() {
            let gpu_prop = unsafe { instance.get_physical_device_properties(*gpu) };
            if gpu_prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                selected_gpu_idx = idx;
                break;
            }
        }
        let (gpu, gfx_qf_idx) = gpu_dets[selected_gpu_idx];
        let queue_priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gfx_qf_idx)
            .queue_priorities(&queue_priorities)];
        let device_extensions = get_device_extensions();
        let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true);
        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features)
            .push_next(&mut device_12_features);
        let device = unsafe {
            instance
                .create_device(gpu, &device_create_info, None)
                .map_err(RhiError::CreateDeviceError)?
        };
        Ok(Self {
            swapchain_device: khr::swapchain::Device::new(&instance, &device),
            device,
            gfx_qf_idx,
            gpu,
            surface,
            surface_instance,
            instance,
            window: window.clone(),
            _entry: entry,
        })
    }

    fn get_surface_formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>, RhiError> {
        unsafe {
            self.surface_instance
                .get_physical_device_surface_formats(self.gpu, self.surface)
                .map_err(RhiError::GetSurfaceFormatsError)
        }
    }

    fn get_surface_caps(&self) -> Result<vk::SurfaceCapabilitiesKHR, RhiError> {
        unsafe {
            self.surface_instance
                .get_physical_device_surface_capabilities(self.gpu, self.surface)
                .map_err(RhiError::GetSurfaceCapsError)
        }
    }

    fn get_surface_present_modes(&self) -> Result<Vec<vk::PresentModeKHR>, RhiError> {
        unsafe {
            self.surface_instance
                .get_physical_device_surface_present_modes(self.gpu, self.surface)
                .map_err(RhiError::GetSurfacePresentModesError)
        }
    }
}

impl Drop for DeviceDropper {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.device.device_wait_idle() {
                warn!("error waiting for device to get idle before destroying: {e}")
            };
            self.device.destroy_device(None);
        }
    }
}

pub struct Device {
    g_queue: Arc<Queue>,
    allocator: Arc<Mutex<Allocator>>,
    inner: Arc<DeviceDropper>,
}

impl Device {
    pub fn new(window: &Arc<Window>) -> Result<Self, RhiError> {
        let device = Arc::new(DeviceDropper::new(window)?);
        let allocator = Arc::new(Mutex::new(Allocator::new(&AllocatorCreateDesc {
            instance: device.instance.clone(),
            device: device.device.clone(),
            physical_device: device.gpu,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?));
        let g_queue = Arc::new(Queue::new(&device)?);
        Ok(Self {
            g_queue,
            allocator,
            inner: device,
        })
    }

    pub fn graphics_queue(&self) -> &Queue {
        &self.g_queue
    }

    pub fn create_swapchain(&self) -> Result<Swapchain, RhiError> {
        Swapchain::new(self)
    }

    pub fn create_buffer(
        &self,
        size: u64,
        usage: BitFlags<BufferFlags>,
        location: MemLocation,
    ) -> Result<Buffer, RhiError> {
        Buffer::new(&self.inner, &self.allocator, size, usage, location)
    }

    pub fn create_image(
        &self,
        dimension: Dimension,
        format: Format,
        width: u32,
        height: u32,
        depth_or_layers: u32,
        mip_levels: u32,
        usage: BitFlags<ImageUsage>,
        location: MemLocation,
    ) -> Result<Image, RhiError> {
        Image::new(
            &self.inner,
            &self.allocator,
            dimension,
            format,
            width,
            height,
            depth_or_layers,
            mip_levels,
            usage,
            location,
        )
    }

    pub fn create_sampler(&self) -> Result<Sampler, RhiError> {
        Sampler::new(&self.inner)
    }

    pub fn create_semaphore(&self, binary: bool) -> Result<Semaphore, RhiError> {
        Semaphore::new(&self.inner, binary)
    }

    pub fn load_shader(&self, code: &[u8]) -> Result<Shader, RhiError> {
        let shader = unsafe {
            self.inner
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(code.align_to().1),
                    None,
                )
                .map_err(RhiError::CreateShaderModuleError)?
        };
        Ok(Shader {
            inner: shader,
            device: self.inner.clone(),
        })
    }

    pub fn create_render_pipeline(
        &self,
        vs_info: VertexStageInfo,
        fs_info: FragmentStageInfo,
        raster_info: RasterMode,
        descriptors: Vec<Vec<DBindingType>>,
        pc_size: u32,
    ) -> Result<RenderPipeline, RhiError> {
        RenderPipeline::new(
            &self.inner,
            vs_info,
            fs_info,
            raster_info,
            descriptors,
            pc_size,
        )
    }
}

const HBR_FORMATS: [Format; 3] = [Format::Rgba16Float, Format::Bgra10, Format::Rgba10];

const SBR_FORMATS: [Format; 2] = [Format::Bgra8, Format::Rgba8];

const COLOR_SPACES: [vk::ColorSpaceKHR; 2] = [
    vk::ColorSpaceKHR::SRGB_NONLINEAR,
    vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
];

fn choose_surface_format(
    surface_formats: &Vec<vk::SurfaceFormatKHR>,
) -> Result<(vk::SurfaceFormatKHR, Format), RhiError> {
    let surface_formats: Vec<_> = surface_formats
        .into_iter()
        .filter(|s| COLOR_SPACES.contains(&s.color_space))
        .collect();
    let surface_format = match HBR_FORMATS.iter().find_map(|format| {
        surface_formats.iter().find_map(|s| {
            if s.format == format.vk() {
                return Some((**s, *format));
            }
            None
        })
    }) {
        Some(sf) => {
            info!(
                "HDR support found. Using colour space {:?} and format {:?}",
                sf.0.color_space, sf.1
            );
            sf
        }
        None => {
            let sf = SBR_FORMATS
                .iter()
                .find_map(|format| {
                    surface_formats.iter().find_map(|s| {
                        if s.format == format.vk() {
                            return Some((**s, *format));
                        }
                        None
                    })
                })
                .ok_or(RhiError::NoSupportedSurfaceFormat)?;
            info!(
                "HDR not supported. Using colour space {:?} and format {:?}",
                sf.0.color_space, sf.1
            );
            sf
        }
    };
    Ok(surface_format)
}

#[derive(Getters, CopyGetters)]
pub struct Swapchain {
    inner: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    fence: vk::Fence,
    #[get = "pub"]
    images: Vec<Image>,
    #[get = "pub"]
    views: Vec<ImageView>,
    #[get_copy = "pub"]
    width: u32,
    #[get_copy = "pub"]
    height: u32,
    queue: Arc<Queue>,
    device: Arc<DeviceDropper>,
}

impl Swapchain {
    fn new(device: &Device) -> Result<Swapchain, RhiError> {
        let surface_formats = device.inner.get_surface_formats()?;
        let surface_caps = device.inner.get_surface_caps()?;
        let surface_present_modes = device.inner.get_surface_present_modes()?;
        let (surface_format, swapchain_format) = choose_surface_format(&surface_formats)?;
        let surface_present_mode = surface_present_modes
            .iter()
            .filter(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .next()
            .cloned()
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let swapchain_image_count = std::cmp::min(
            surface_caps.min_image_count + 1,
            if surface_caps.max_image_count == 0 {
                std::u32::MAX
            } else {
                surface_caps.max_image_count
            },
        );
        let mut surface_resolution = surface_caps.current_extent;
        if surface_resolution.width == u32::MAX || surface_resolution.height == u32::MAX {
            let window_res = device.inner.window.inner_size();
            surface_resolution.width = window_res.width;
            surface_resolution.height = window_res.height;
        }
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(device.inner.surface)
            .min_image_count(swapchain_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_resolution)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface_present_mode)
            .clipped(true);
        let swapchain = unsafe {
            device
                .inner
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(RhiError::CreateSwapchainError)?
        };
        let images: Vec<_> = unsafe {
            device
                .inner
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(RhiError::GetSwapchainImagesError)?
                .into_iter()
                .map(|img| Image {
                    inner: Arc::new(ImageDropper {
                        inner: img,
                        device: device.inner.clone(),
                        memory: None,
                    }),
                    format: swapchain_format,
                    dimension: Dimension::D2,
                    width: surface_resolution.width,
                    height: surface_resolution.height,
                    depth: 1,
                    mip_levels: 1,
                })
                .collect()
        };
        let views: Vec<_> = images
            .iter()
            .map(|img| img.create_view(ViewDimension::D2, 0..1, 0..1))
            .collect::<Result<_, _>>()?;
        let fence = unsafe {
            device
                .inner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(RhiError::CreateFenceError)?
        };
        Ok(Self {
            inner: swapchain,
            format: surface_format,
            present_mode: surface_present_mode,
            fence,
            images,
            views,
            width: surface_resolution.width,
            height: surface_resolution.height,
            queue: device.g_queue.clone(),
            device: device.inner.clone(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RhiError> {
        let current_transform = self
            .device
            .get_surface_caps()
            .map(|c| c.current_transform)
            .unwrap_or(vk::SurfaceTransformFlagsKHR::IDENTITY);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.device.surface)
            .min_image_count(self.images.len() as _)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(vk::Extent2D { width, height })
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .pre_transform(current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(self.inner);
        let swapchain = unsafe {
            self.device
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(RhiError::CreateSwapchainError)?
        };
        let images: Vec<_> = unsafe {
            self.device
                .swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(RhiError::GetSwapchainImagesError)?
                .into_iter()
                .map(|img| Image {
                    inner: Arc::new(ImageDropper {
                        inner: img,
                        device: self.device.clone(),
                        memory: None,
                    }),
                    format: self.images[0].format,
                    dimension: Dimension::D2,
                    width,
                    height,
                    depth: 1,
                    mip_levels: 1,
                })
                .collect()
        };
        let views: Vec<_> = images
            .iter()
            .map(|img| img.create_view(ViewDimension::D2, 0..1, 0..1))
            .collect::<Result<_, _>>()?;

        self.views = views;
        self.images = images;
        unsafe {
            self.device
                .swapchain_device
                .destroy_swapchain(self.inner, None);
        }
        self.inner = swapchain;
        Ok(())
    }

    pub fn acquire_image(&self) -> Result<(u32, bool), RhiError> {
        unsafe {
            let (idx, outdated) = self
                .device
                .swapchain_device
                .acquire_next_image(self.inner, u64::MAX, vk::Semaphore::null(), self.fence)
                .map_err(RhiError::AcquireSwapchainImageError)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(RhiError::WaitFenceError)?;
            self.device
                .device
                .reset_fences(&[self.fence])
                .map_err(RhiError::ResetFenceError)?;
            Ok((idx, outdated))
        }
    }

    pub fn present_image(&self, idx: u32, semaphore: &Semaphore) -> Result<bool, RhiError> {
        if !semaphore.is_binary {
            return Err(RhiError::UnsupportedSemaphoreType);
        }
        unsafe {
            self.device
                .swapchain_device
                .queue_present(
                    self.queue.cmd_pool.queue,
                    &vk::PresentInfoKHR::default()
                        .swapchains(&[self.inner])
                        .image_indices(&[idx])
                        .wait_semaphores(&[semaphore.inner]),
                )
                .map_err(RhiError::PresentSwapchainImageError)
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.device
                .swapchain_device
                .destroy_swapchain(self.inner, None);
            self.device.device.destroy_fence(self.fence, None);
        }
    }
}

struct CommandPoolDropper {
    inner: vk::CommandPool,
    qf_idx: u32,
    queue: vk::Queue,
    device: Arc<DeviceDropper>,
}

impl Drop for CommandPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_command_pool(self.inner, None);
        }
    }
}

pub struct Queue {
    cmd_pool: Arc<CommandPoolDropper>,
    device: Arc<DeviceDropper>,
}

impl Queue {
    fn new(device: &Arc<DeviceDropper>) -> Result<Self, RhiError> {
        let queue = unsafe { device.device.get_device_queue(device.gfx_qf_idx, 0) };
        let cmd_pool = unsafe {
            device
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(device.gfx_qf_idx)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(RhiError::CreateCommandPoolError)?
        };
        let cmd_pool = Arc::new(CommandPoolDropper {
            inner: cmd_pool,
            qf_idx: device.gfx_qf_idx,
            queue,
            device: device.clone(),
        });
        Ok(Self {
            cmd_pool,
            device: device.clone(),
        })
    }

    pub fn create_command_buffer(&self) -> Result<CommandBuffer, RhiError> {
        let cmd_buffer = unsafe {
            self.device
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool.inner)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(RhiError::CreateCommandBufferError)?[0]
        };
        Ok(CommandBuffer {
            inner: cmd_buffer,
            command_pool: self.cmd_pool.clone(),
        })
    }

    pub fn wait_idle(&self) {
        unsafe {
            if let Err(e) = self.device.device.queue_wait_idle(self.cmd_pool.queue) {
                warn!("error while waiting for queue to be idle: {e}")
            }
        }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.wait_idle();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MemLocation {
    Gpu,
    CpuToGpu,
    GpuToCpu,
}

impl MemLocation {
    fn gpu_alloc(&self) -> MemoryLocation {
        match self {
            MemLocation::Gpu => MemoryLocation::GpuOnly,
            MemLocation::CpuToGpu => MemoryLocation::CpuToGpu,
            MemLocation::GpuToCpu => MemoryLocation::GpuToCpu,
        }
    }
}

struct Memory {
    name: String,
    inner: ManuallyDrop<Allocation>,
    altr: Arc<Mutex<Allocator>>,
}

impl Memory {
    fn new(
        altr: &Arc<Mutex<Allocator>>,
        name: String,
        requirements: vk::MemoryRequirements,
        location: MemLocation,
        linear: bool,
    ) -> Result<Self, RhiError> {
        let mut altr_guard = match altr.lock() {
            Ok(a) => a,
            Err(e) => {
                warn!("allocator lock found poisoned:{e}");
                e.into_inner()
            }
        };
        let altn = altr_guard.allocate(&AllocationCreateDesc {
            name: &name,
            requirements,
            location: location.gpu_alloc(),
            linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        Ok(Self {
            name,
            inner: ManuallyDrop::new(altn),
            altr: altr.clone(),
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        let mut altr = match self.altr.lock() {
            Ok(a) => a,
            Err(e) => {
                warn!("allocator lock found poisoned:{e}");
                e.into_inner()
            }
        };
        unsafe {
            let mem = ManuallyDrop::take(&mut self.inner);
            if let Err(e) = altr.free(mem) {
                warn!("error freeing memory of allocation {}: {e}", &self.name)
            }
        }
    }
}

#[bitflags]
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BufferFlags {
    CopySrc,
    CopyDst,
    Vertex,
    Index,
    Uniform,
    Storage,
}

impl BufferFlags {
    fn to_vk(flags: BitFlags<Self>) -> vk::BufferUsageFlags {
        let mut out = vk::BufferUsageFlags::empty();
        for bit in flags.iter() {
            match bit {
                Self::CopySrc => out |= vk::BufferUsageFlags::TRANSFER_SRC,
                Self::CopyDst => out |= vk::BufferUsageFlags::TRANSFER_DST,
                Self::Vertex => out |= vk::BufferUsageFlags::VERTEX_BUFFER,
                Self::Index => out |= vk::BufferUsageFlags::INDEX_BUFFER,
                Self::Uniform => out |= vk::BufferUsageFlags::UNIFORM_BUFFER,
                Self::Storage => out |= vk::BufferUsageFlags::STORAGE_BUFFER,
            }
        }
        out
    }
}

pub struct Buffer {
    inner: vk::Buffer,
    size: u64,
    memory: Memory,
    device: Arc<DeviceDropper>,
}

impl Buffer {
    fn new(
        device: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        size: u64,
        usage: BitFlags<BufferFlags>,
        location: MemLocation,
    ) -> Result<Self, RhiError> {
        let usage = BufferFlags::to_vk(usage);
        let buffer = unsafe {
            device
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().usage(usage).size(size),
                    None,
                )
                .map_err(RhiError::CreateBufferError)?
        };
        let mem_req = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let memory = Memory::new(
            &allocator,
            format!("buffer_{:?}", buffer),
            mem_req,
            location,
            true,
        )?;
        unsafe {
            device
                .device
                .bind_buffer_memory(buffer, memory.inner.memory(), memory.inner.offset())
                .map_err(RhiError::BufferBindMemError)?;
        }
        Ok(Self {
            inner: buffer,
            memory,
            size,
            device: device.clone(),
        })
    }

    pub fn write_data(&mut self, data: &[u8]) -> Result<(), RhiError> {
        self.memory
            .inner
            .mapped_slice_mut()
            .ok_or(RhiError::MemReadOnly)?
            .copy_from_slice(data);
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.inner, None);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    D1,
    D2,
    D3,
}

impl Dimension {
    fn vk(&self) -> vk::ImageType {
        match self {
            Dimension::D1 => vk::ImageType::TYPE_1D,
            Dimension::D2 => vk::ImageType::TYPE_2D,
            Dimension::D3 => vk::ImageType::TYPE_3D,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Rgba8,
    Bgra8,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba10,
    Bgra10,
    Rgba16,
    Rgba16Float,
    D24S8,
    D32Float,
}

impl Format {
    fn vk(&self) -> vk::Format {
        match self {
            Format::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            Format::Bgra8 => vk::Format::B8G8R8A8_UNORM,
            Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            Format::Rgba10 => vk::Format::A2R10G10B10_UNORM_PACK32,
            Format::Bgra10 => vk::Format::A2B10G10R10_UNORM_PACK32,
            Format::Rgba16 => vk::Format::R16G16B16A16_UNORM,
            Format::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Format::D24S8 => vk::Format::D24_UNORM_S8_UINT,
            Format::D32Float => vk::Format::D32_SFLOAT,
        }
    }

    fn is_depth_sencil(&self) -> (bool, bool) {
        match self {
            Format::D24S8 => (true, true),
            Format::D32Float => (true, false),
            _ => (false, false),
        }
    }

    fn aspect_flag(&self) -> vk::ImageAspectFlags {
        let (depth, stencil) = Self::is_depth_sencil(&self);
        if depth {
            if stencil {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        } else {
            vk::ImageAspectFlags::COLOR
        }
    }
}

#[bitflags]
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ImageUsage {
    CopySrc,
    CopyDst,
    Sampled,
    Storage,
    Attachment,
}

impl ImageUsage {
    fn vk(flags: BitFlags<Self>, format: Format) -> vk::ImageUsageFlags {
        let mut out = vk::ImageUsageFlags::empty();
        for bit in flags.iter() {
            match bit {
                Self::CopySrc => out |= vk::ImageUsageFlags::TRANSFER_SRC,
                Self::CopyDst => out |= vk::ImageUsageFlags::TRANSFER_DST,
                Self::Sampled => out |= vk::ImageUsageFlags::SAMPLED,
                Self::Storage => out |= vk::ImageUsageFlags::STORAGE,
                Self::Attachment => {
                    let (depth, _) = format.is_depth_sencil();
                    out |= if depth {
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    } else {
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                    }
                }
            }
        }
        out
    }
}

struct ImageDropper {
    inner: vk::Image,
    device: Arc<DeviceDropper>,
    memory: Option<Memory>,
}

impl Drop for ImageDropper {
    fn drop(&mut self) {
        unsafe {
            if self.memory.is_some() {
                self.device.device.destroy_image(self.inner, None);
            }
        }
    }
}

#[derive(Getters, CopyGetters, Clone)]
pub struct Image {
    inner: Arc<ImageDropper>,
    #[get_copy = "pub"]
    format: Format,
    #[get_copy = "pub"]
    dimension: Dimension,
    #[get_copy = "pub"]
    width: u32,
    #[get_copy = "pub"]
    height: u32,
    #[get_copy = "pub"]
    depth: u32,
    #[get_copy = "pub"]
    mip_levels: u32,
}

impl Image {
    fn new(
        device: &Arc<DeviceDropper>,
        allocator: &Arc<Mutex<Allocator>>,
        dimension: Dimension,
        format: Format,
        width: u32,
        height: u32,
        depth_or_layers: u32,
        mip_levels: u32,
        usage: BitFlags<ImageUsage>,
        location: MemLocation,
    ) -> Result<Self, RhiError> {
        let (extent, layers) = if dimension == Dimension::D3 {
            let extent = vk::Extent3D {
                width,
                height,
                depth: depth_or_layers,
            };
            (extent, 1)
        } else {
            let extent = vk::Extent3D {
                width,
                height,
                depth: 1,
            };
            (extent, depth_or_layers)
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(dimension.vk())
            .format(format.vk())
            .samples(vk::SampleCountFlags::TYPE_1)
            .extent(extent)
            .array_layers(layers)
            .initial_layout(ImageAccess::Undefined.layout(format))
            .mip_levels(mip_levels)
            .usage(ImageUsage::vk(usage, format));
        let image = unsafe {
            device
                .device
                .create_image(&create_info, None)
                .map_err(RhiError::CreateImageError)?
        };
        let mem_req = unsafe { device.device.get_image_memory_requirements(image) };
        let memory = Memory::new(
            &allocator,
            format!("image_{:?}", image),
            mem_req,
            location,
            true,
        )?;
        unsafe {
            device
                .device
                .bind_image_memory(image, memory.inner.memory(), memory.inner.offset())
                .map_err(RhiError::ImageBindMemError)?;
        }
        Ok(Self {
            inner: Arc::new(ImageDropper {
                inner: image,
                device: device.clone(),
                memory: Some(memory),
            }),
            format,
            dimension,
            width,
            height,
            depth: depth_or_layers,
            mip_levels,
        })
    }

    pub fn create_view(
        &self,
        view_dim: ViewDimension,
        layer_range: Range<u32>,
        mip_level_range: Range<u32>,
    ) -> Result<ImageView, RhiError> {
        let layer_range = if let Dimension::D3 = self.dimension {
            Range { start: 0, end: 1 }
        } else {
            Range {
                start: layer_range.start.min(self.depth),
                end: layer_range.end.max(self.depth),
            }
        };
        let mip_level_range = Range {
            start: mip_level_range.start.min(self.mip_levels),
            end: mip_level_range.end.min(self.mip_levels),
        };
        let create_info = vk::ImageViewCreateInfo::default()
            .view_type(view_dim.vk(&layer_range))
            .format(self.format.vk())
            .image(self.inner.inner)
            .components(vk::ComponentMapping::default())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(self.format.aspect_flag())
                    .base_mip_level(mip_level_range.start)
                    .level_count(mip_level_range.len() as _)
                    .base_array_layer(layer_range.start)
                    .layer_count(layer_range.len() as _),
            );
        let view = unsafe {
            self.inner
                .device
                .device
                .create_image_view(&create_info, None)
                .map_err(RhiError::CreateImageViewError)?
        };
        Ok(ImageView {
            _dimension: view_dim,
            dropper: ImageViewDropper {
                inner: view,
                image: self.clone(),
            }
            .into(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ViewDimension {
    D1,
    D2,
    D3,
    Cube,
}

impl ViewDimension {
    fn vk(&self, layer_range: &Range<u32>) -> vk::ImageViewType {
        let is_array = layer_range.len() > 1;
        if is_array {
            match self {
                ViewDimension::D1 => vk::ImageViewType::TYPE_1D,
                ViewDimension::D2 => vk::ImageViewType::TYPE_2D,
                ViewDimension::D3 => vk::ImageViewType::TYPE_3D,
                ViewDimension::Cube => vk::ImageViewType::CUBE,
            }
        } else {
            match self {
                ViewDimension::D1 => vk::ImageViewType::TYPE_1D_ARRAY,
                ViewDimension::D2 => vk::ImageViewType::TYPE_2D_ARRAY,
                ViewDimension::D3 => vk::ImageViewType::TYPE_3D,
                ViewDimension::Cube => vk::ImageViewType::CUBE_ARRAY,
            }
        }
    }
}

pub struct ImageViewDropper {
    inner: vk::ImageView,
    image: Image,
}

impl Drop for ImageViewDropper {
    fn drop(&mut self) {
        unsafe {
            self.image
                .inner
                .device
                .device
                .destroy_image_view(self.inner, None);
        }
    }
}

#[derive(Clone)]
pub struct ImageView {
    dropper: Arc<ImageViewDropper>,
    _dimension: ViewDimension,
}

#[derive(Debug, Clone, Copy)]
pub enum RWAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageAccess {
    Undefined,
    Transfer(RWAccess),
    Shader(RWAccess),
    Attachment(RWAccess),
    Present,
}

impl ImageAccess {
    fn stage(&self) -> vk::PipelineStageFlags {
        match self {
            ImageAccess::Undefined => vk::PipelineStageFlags::ALL_COMMANDS,
            ImageAccess::Transfer(_) => vk::PipelineStageFlags::TRANSFER,
            ImageAccess::Shader(_) => vk::PipelineStageFlags::FRAGMENT_SHADER,
            ImageAccess::Attachment(_) => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ImageAccess::Present => vk::PipelineStageFlags::ALL_COMMANDS,
        }
    }

    fn layout(&self, format: Format) -> vk::ImageLayout {
        let (depth, _stencil) = format.is_depth_sencil();
        match self {
            ImageAccess::Undefined => vk::ImageLayout::UNDEFINED,
            ImageAccess::Transfer(rwaccess) => match rwaccess {
                RWAccess::Read => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                RWAccess::Write => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                RWAccess::ReadWrite => {
                    warn!("can't use both read and write transfer layout");
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                }
            },
            ImageAccess::Shader(_) => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageAccess::Attachment(_) => {
                if depth {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                }
            }

            ImageAccess::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    fn access(&self, format: Format) -> vk::AccessFlags {
        let (depth, _stencil) = format.is_depth_sencil();
        match self {
            ImageAccess::Undefined => vk::AccessFlags::empty(),
            ImageAccess::Transfer(rwaccess) => match rwaccess {
                RWAccess::Read => vk::AccessFlags::TRANSFER_READ,
                RWAccess::Write => vk::AccessFlags::TRANSFER_WRITE,
                RWAccess::ReadWrite => {
                    vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE
                }
            },
            ImageAccess::Shader(rwaccess) => match rwaccess {
                RWAccess::Read => vk::AccessFlags::SHADER_READ,
                RWAccess::Write => vk::AccessFlags::SHADER_WRITE,
                RWAccess::ReadWrite => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            },
            ImageAccess::Attachment(rwaccess) => {
                if depth {
                    match rwaccess {
                        RWAccess::Read => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        RWAccess::Write => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        RWAccess::ReadWrite => {
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                        }
                    }
                } else {
                    match rwaccess {
                        RWAccess::Read => vk::AccessFlags::COLOR_ATTACHMENT_READ,
                        RWAccess::Write => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        RWAccess::ReadWrite => {
                            vk::AccessFlags::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        }
                    }
                }
            }

            ImageAccess::Present => vk::AccessFlags::empty(),
        }
    }
}

pub struct Sampler {
    inner: vk::Sampler,
    device: Arc<DeviceDropper>,
}

impl Sampler {
    fn new(device: &Arc<DeviceDropper>) -> Result<Sampler, RhiError> {
        let sampler = unsafe {
            device
                .device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .map_err(RhiError::CreateSamplerError)?
        };

        Ok(Self {
            inner: sampler,
            device: device.clone(),
        })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.inner, None);
        }
    }
}

pub struct CommandBuffer {
    inner: vk::CommandBuffer,
    command_pool: Arc<CommandPoolDropper>,
}

impl CommandBuffer {
    pub fn encoder(&self) -> Result<CommandEncoder, RhiError> {
        unsafe {
            self.command_pool
                .device
                .device
                .begin_command_buffer(self.inner, &vk::CommandBufferBeginInfo::default())
                .map_err(RhiError::BeginCommandBufferError)?;
        }
        Ok(CommandEncoder {
            last_image_access: HashMap::new(),
            cmd_buffer: self.inner,
            cmd_pool: self.command_pool.clone(),
        })
    }

    pub fn submit(
        &self,
        wait: Vec<SemSubmitInfo>,
        emit: Vec<SemSubmitInfo>,
    ) -> Result<(), RhiError> {
        let wait_values: Vec<_> = wait.iter().map(|s| s.num).collect();
        let emit_values: Vec<_> = emit.iter().map(|s| s.num).collect();
        let wait_sems: Vec<_> = wait.iter().map(|s| s.sem).collect();
        let emit_sems: Vec<_> = emit.iter().map(|s| s.sem).collect();
        let mut tl_sem_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&emit_values);
        unsafe {
            self.command_pool
                .device
                .device
                .queue_submit(
                    self.command_pool.queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[self.inner])
                        .wait_semaphores(&wait_sems)
                        .signal_semaphores(&emit_sems)
                        .push_next(&mut tl_sem_info)],
                    vk::Fence::null(),
                )
                .map_err(RhiError::SubmitCommandBufferError)?;
        }
        Ok(())
    }
}

pub struct CommandEncoder {
    last_image_access: HashMap<vk::Image, ImageAccess>,
    cmd_buffer: vk::CommandBuffer,
    cmd_pool: Arc<CommandPoolDropper>,
}

impl CommandEncoder {
    pub fn set_last_image_access(
        &mut self,
        image: &Image,
        access: ImageAccess,
        layer_range: Range<u32>,
        mip_level_range: Range<u32>,
    ) {
        if let Some(last_access) = self.last_image_access.insert(image.inner.inner, access) {
            unsafe {
                self.cmd_pool.device.device.cmd_pipeline_barrier(
                    self.cmd_buffer,
                    last_access.stage(),
                    access.stage(),
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(image.inner.inner)
                        .old_layout(last_access.layout(image.format))
                        .new_layout(access.layout(image.format))
                        .src_access_mask(last_access.access(image.format))
                        .dst_access_mask(access.access(image.format))
                        .src_queue_family_index(self.cmd_pool.qf_idx)
                        .dst_queue_family_index(self.cmd_pool.qf_idx)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(image.format.aspect_flag())
                                .base_mip_level(mip_level_range.start)
                                .level_count(mip_level_range.len() as _)
                                .base_array_layer(layer_range.start)
                                .layer_count(layer_range.len() as _),
                        )],
                );
            }
        }
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, dst: &Buffer) {
        unsafe {
            self.cmd_pool.device.device.cmd_copy_buffer(
                self.cmd_buffer,
                src.inner,
                dst.inner,
                &[vk::BufferCopy::default().size(src.size.min(dst.size))],
            );
        }
    }

    pub fn copy_buffer_to_image(
        &mut self,
        buffer: &Buffer,
        image: &Image,
        layer_range: Range<u32>,
        mip_level: u32,
    ) {
        self.set_last_image_access(
            image,
            ImageAccess::Transfer(RWAccess::Write),
            layer_range.clone(),
            mip_level..mip_level + 1,
        );
        unsafe {
            self.cmd_pool.device.device.cmd_copy_buffer_to_image(
                self.cmd_buffer,
                buffer.inner,
                image.inner.inner,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(image.format.aspect_flag())
                            .base_array_layer(layer_range.start)
                            .layer_count(layer_range.len() as _)
                            .mip_level(mip_level),
                    )
                    .image_extent(vk::Extent3D {
                        width: image.width,
                        height: image.height,
                        depth: image.depth,
                    })],
            );
        }
    }

    pub fn blit_image(
        &mut self,
        src: &Image,
        dst: &Image,
        src_layer_range: Range<u32>,
        dst_layer_range: Range<u32>,
        src_mip_level: u32,
        dst_mip_level: u32,
        src_range: [[f32; 3]; 2],
        dst_range: [[f32; 3]; 2],
    ) {
        self.set_last_image_access(
            src,
            ImageAccess::Transfer(RWAccess::Read),
            src_layer_range.clone(),
            src_mip_level..src_mip_level + 1,
        );
        self.set_last_image_access(
            dst,
            ImageAccess::Transfer(RWAccess::Write),
            dst_layer_range.clone(),
            dst_mip_level..dst_mip_level + 1,
        );
        let src_offsets = [
            vk::Offset3D {
                x: (src_range[0][0] * src.width as f32) as _,
                y: (src_range[0][1] * src.height as f32) as _,
                z: if src.dimension == Dimension::D3 {
                    (src_range[0][2] * src.depth as f32) as _
                } else {
                    0
                },
            },
            vk::Offset3D {
                x: (src_range[1][0] * src.width as f32) as _,
                y: (src_range[1][1] * src.height as f32) as _,
                z: if src.dimension == Dimension::D3 {
                    (src_range[1][2] * src.depth as f32) as _
                } else {
                    1
                },
            },
        ];
        let dst_offsets = [
            vk::Offset3D {
                x: (dst_range[0][0] * dst.width as f32) as _,
                y: (dst_range[0][1] * dst.height as f32) as _,
                z: if dst.dimension == Dimension::D3 {
                    (dst_range[0][2] * dst.depth as f32) as _
                } else {
                    0
                },
            },
            vk::Offset3D {
                x: (dst_range[1][0] * dst.width as f32) as _,
                y: (dst_range[1][1] * dst.height as f32) as _,
                z: if dst.dimension == Dimension::D3 {
                    (dst_range[1][2] * dst.depth as f32) as _
                } else {
                    1
                },
            },
        ];
        unsafe {
            self.cmd_pool.device.device.cmd_blit_image(
                self.cmd_buffer,
                src.inner.inner,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.inner.inner,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_offsets(src_offsets)
                    .dst_offsets(dst_offsets)
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(src.format.aspect_flag())
                            .base_array_layer(src_layer_range.start)
                            .layer_count(src_layer_range.len() as _)
                            .mip_level(src_mip_level),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(dst.format.aspect_flag())
                            .base_array_layer(dst_layer_range.start)
                            .layer_count(dst_layer_range.len() as _)
                            .mip_level(dst_mip_level),
                    )],
                vk::Filter::NEAREST,
            );
        }
    }

    pub fn blit_image_2d_stretch(
        &mut self,
        src: &Image,
        dst: &Image,
        src_mip_level: u32,
        dst_mip_level: u32,
    ) {
        self.blit_image(
            src,
            dst,
            0..1,
            0..1,
            src_mip_level,
            dst_mip_level,
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        );
    }

    pub fn start_render_pipeline(
        mut self,
        pipeline: &RenderPipeline,
        output: &RenderOutput,
        clear_values: Vec<ClearValue>,
    ) -> RenderCommandEncoder {
        for img in &output.images {
            self.set_last_image_access(
                &img.dropper.image,
                ImageAccess::Attachment(RWAccess::ReadWrite),
                0..1,
                0..1,
            );
        }
        unsafe {
            self.cmd_pool.device.device.cmd_begin_render_pass(
                self.cmd_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pipeline.render_pass)
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D { x: 0, y: 0 })
                            .extent(vk::Extent2D {
                                width: output.images[0].dropper.image.width as _,
                                height: output.images[0].dropper.image.height as _,
                            }),
                    )
                    .framebuffer(output.inner)
                    .clear_values(&clear_values.iter().map(|c| c.vk()).collect::<Vec<_>>()),
                vk::SubpassContents::INLINE,
            );
            self.cmd_pool.device.device.cmd_bind_pipeline(
                self.cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            self.cmd_pool.device.device.cmd_set_viewport(
                self.cmd_buffer,
                0,
                &[vk::Viewport::default()
                    .x(0.0)
                    .y(output.images[0].dropper.image.height as _)
                    .width(output.images[0].dropper.image.width as _)
                    .height(-(output.images[0].dropper.image.height as f32))
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            self.cmd_pool.device.device.cmd_set_scissor(
                self.cmd_buffer,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(vk::Extent2D {
                        width: output.images[0].dropper.image.width,
                        height: output.images[0].dropper.image.height,
                    })],
            );
        }
        RenderCommandEncoder {
            encoder: self,
            _render_pass: pipeline.render_pass,
            _pipeline: pipeline.pipeline,
            layout: pipeline.pipeline_layout,
            _framebuffer: output.inner,
        }
    }

    pub fn finalize(self) -> Result<(), RhiError> {
        unsafe {
            self.cmd_pool
                .device
                .device
                .end_command_buffer(self.cmd_buffer)
                .map_err(RhiError::EndCommandBufferError)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    // U8,
    U16,
    U32,
}

impl IndexType {
    fn vk(&self) -> vk::IndexType {
        match self {
            // IndexType::U8 => vk::IndexType::UINT8_KHR,
            IndexType::U16 => vk::IndexType::UINT16,
            IndexType::U32 => vk::IndexType::UINT32,
        }
    }
}

pub struct RenderCommandEncoder {
    encoder: CommandEncoder,
    _render_pass: vk::RenderPass,
    _pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    _framebuffer: vk::Framebuffer,
}

impl RenderCommandEncoder {
    pub fn bind_vbs(&mut self, vbs: Vec<&Buffer>) {
        unsafe {
            self.encoder.cmd_pool.device.device.cmd_bind_vertex_buffers(
                self.encoder.cmd_buffer,
                0,
                &vbs.iter().map(|b| b.inner).collect::<Vec<_>>(),
                &vec![0; vbs.len()],
            );
        }
    }

    pub fn bind_ib(&mut self, ib: &Buffer, it: IndexType) {
        unsafe {
            self.encoder.cmd_pool.device.device.cmd_bind_index_buffer(
                self.encoder.cmd_buffer,
                ib.inner,
                0,
                it.vk(),
            );
        }
    }

    pub fn bind_dsets(&mut self, dset: Vec<&DSet>) {
        let sets_vk: Vec<_> = dset.iter().map(|d| d.inner).collect();
        unsafe {
            self.encoder
                .cmd_pool
                .device
                .device
                .cmd_bind_descriptor_sets(
                    self.encoder.cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &sets_vk,
                    &[],
                );
        }
    }

    pub fn draw_indexed(&mut self, count: u32) {
        unsafe {
            self.encoder.cmd_pool.device.device.cmd_draw_indexed(
                self.encoder.cmd_buffer,
                count,
                1,
                0,
                0,
                0,
            );
        }
    }

    pub fn draw(&mut self, count: u32) {
        unsafe {
            self.encoder
                .cmd_pool
                .device
                .device
                .cmd_draw(self.encoder.cmd_buffer, count, 1, 0, 0);
        }
    }

    pub fn end(self) -> CommandEncoder {
        unsafe {
            self.encoder
                .cmd_pool
                .device
                .device
                .cmd_end_render_pass(self.encoder.cmd_buffer);
        }
        self.encoder
    }
}

pub struct SemSubmitInfo {
    sem: vk::Semaphore,
    num: u64,
    _is_binary: bool,
}

pub struct Semaphore {
    inner: vk::Semaphore,
    is_binary: bool,
    device: Arc<DeviceDropper>,
}

impl Semaphore {
    fn new(device: &Arc<DeviceDropper>, binary: bool) -> Result<Self, RhiError> {
        let mut sem_type_info = if binary {
            vk::SemaphoreTypeCreateInfo::default()
                .initial_value(0)
                .semaphore_type(vk::SemaphoreType::BINARY)
        } else {
            vk::SemaphoreTypeCreateInfo::default()
                .initial_value(0)
                .semaphore_type(vk::SemaphoreType::TIMELINE)
        };
        let semaphore = unsafe {
            device
                .device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut sem_type_info),
                    None,
                )
                .map_err(RhiError::CreateSemaphoreError)?
        };
        Ok(Self {
            inner: semaphore,
            is_binary: binary,
            device: device.clone(),
        })
    }
    pub fn submit_info(&self, num: u64) -> SemSubmitInfo {
        SemSubmitInfo {
            sem: self.inner,
            num,
            _is_binary: self.is_binary,
        }
    }

    pub fn wait_for(&self, num: u64, timeout: Option<u64>) -> Result<(), RhiError> {
        let timeout = timeout.unwrap_or(u64::MAX);
        unsafe {
            self.device
                .device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.inner])
                        .values(&[num]),
                    timeout,
                )
                .map_err(RhiError::WaitSemaphoreError)?;
        }
        Ok(())
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_semaphore(self.inner, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DBindingType {
    UBuffer(u32),
    SBuffer(u32),
    Sampler2d(u32),
}

impl DBindingType {
    fn vk(&self) -> vk::DescriptorType {
        match self {
            DBindingType::UBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            DBindingType::SBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            DBindingType::Sampler2d(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }

    fn count(&self) -> u32 {
        match self {
            DBindingType::UBuffer(c) => *c,
            DBindingType::SBuffer(c) => *c,
            DBindingType::Sampler2d(c) => *c,
        }
    }
}

struct DPoolDropper {
    pool: vk::DescriptorPool,
    device: Arc<DeviceDropper>,
}

impl Drop for DPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

struct DAlloc {
    dsl: vk::DescriptorSetLayout,
    bindings: Vec<DBindingType>,
    pool: Arc<DPoolDropper>,
}

impl DAlloc {
    fn new(device: &Arc<DeviceDropper>, bindings: &[DBindingType]) -> Result<Self, RhiError> {
        let binding_vk: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as _)
                    .descriptor_type(b.vk())
                    .descriptor_count(b.count())
                    .stage_flags(vk::ShaderStageFlags::ALL)
            })
            .collect();
        let dsl_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&binding_vk)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
        let dsl = unsafe {
            device
                .device
                .create_descriptor_set_layout(&dsl_create_info, None)
                .map_err(RhiError::CreateDslError)?
        };
        let pool_sizes: Vec<_> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorPoolSize::default()
                    .ty(b.vk())
                    .descriptor_count(b.count() * 128)
            })
            .collect();
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&pool_sizes)
            .max_sets(128);
        let pool = unsafe {
            device
                .device
                .create_descriptor_pool(&pool_create_info, None)
                .map_err(RhiError::CreateDPoolError)?
        };
        let pool = DPoolDropper {
            pool,
            device: device.clone(),
        };
        Ok(Self {
            dsl,
            bindings: bindings.to_vec(),
            pool: Arc::new(pool),
        })
    }

    fn new_set(&mut self) -> Result<DSet, RhiError> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool.pool)
            .set_layouts(core::slice::from_ref(&self.dsl));
        let dset = unsafe {
            match self
                .pool
                .device
                .device
                .allocate_descriptor_sets(&alloc_info)
            {
                Ok(mut dset) => dset.remove(0),
                Err(e) => match e {
                    vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                        let pool_sizes: Vec<_> = self
                            .bindings
                            .iter()
                            .map(|b| {
                                vk::DescriptorPoolSize::default()
                                    .ty(b.vk())
                                    .descriptor_count(b.count() * 128)
                            })
                            .collect();
                        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                            .pool_sizes(&pool_sizes)
                            .max_sets(128);
                        let pool = self
                            .pool
                            .device
                            .device
                            .create_descriptor_pool(&pool_create_info, None)
                            .map_err(RhiError::CreateDPoolError)?;
                        let pool = DPoolDropper {
                            pool,
                            device: self.pool.device.clone(),
                        };
                        self.pool = Arc::new(pool);
                        let dset = self
                            .pool
                            .device
                            .device
                            .allocate_descriptor_sets(&alloc_info)
                            .map_err(RhiError::CreateDSetError)?
                            .remove(0);
                        dset
                    }
                    _ => return Err(RhiError::CreateDSetError(e)),
                },
            }
        };
        let dset = DSet {
            inner: dset,
            pool: self.pool.clone(),
        };
        Ok(dset)
    }
}

impl Drop for DAlloc {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device
                .device
                .destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}

pub enum DBindingData<'a> {
    UBuffer(Vec<&'a Buffer>),
    SBuffer(Vec<&'a Buffer>),
    Sampler2d(Vec<(&'a ImageView, &'a Sampler)>),
}

impl<'a> DBindingData<'a> {
    fn vk_type(&self) -> vk::DescriptorType {
        match self {
            DBindingData::UBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            DBindingData::SBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            DBindingData::Sampler2d(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }

    fn count(&self) -> u32 {
        match self {
            DBindingData::UBuffer(buffers) => buffers.len() as _,
            DBindingData::SBuffer(buffers) => buffers.len() as _,
            DBindingData::Sampler2d(items) => items.len() as _,
        }
    }

    fn vk_infos(&self) -> (Vec<vk::DescriptorBufferInfo>, Vec<vk::DescriptorImageInfo>) {
        match self {
            DBindingData::UBuffer(buffers) => (
                buffers
                    .iter()
                    .map(|b| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(b.inner)
                            .range(b.size)
                    })
                    .collect(),
                vec![],
            ),
            DBindingData::SBuffer(buffers) => (
                buffers
                    .iter()
                    .map(|b| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(b.inner)
                            .range(b.size)
                    })
                    .collect(),
                vec![],
            ),
            DBindingData::Sampler2d(items) => (
                vec![],
                items
                    .iter()
                    .map(|(img, sam)| {
                        vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(img.dropper.inner)
                            .sampler(sam.inner)
                    })
                    .collect(),
            ),
        }
    }
}

pub struct DSet {
    inner: vk::DescriptorSet,
    pool: Arc<DPoolDropper>,
}

impl DSet {
    pub fn write(&mut self, data: Vec<DBindingData>) {
        let (b_infos, i_infos): (Vec<_>, Vec<_>) = data.iter().map(|b| b.vk_infos()).unzip();
        let writes: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let mut w = vk::WriteDescriptorSet::default()
                    .dst_set(self.inner)
                    .dst_binding(i as _)
                    .descriptor_type(b.vk_type())
                    .descriptor_count(b.count());
                if !b_infos[i].is_empty() {
                    w = w.buffer_info(&b_infos[i]);
                }
                if !i_infos[i].is_empty() {
                    w = w.image_info(&i_infos[i]);
                }
                w
            })
            .collect();
        unsafe {
            self.pool.device.device.update_descriptor_sets(&writes, &[]);
        }
    }
}

pub struct Shader {
    inner: vk::ShaderModule,
    device: Arc<DeviceDropper>,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_shader_module(self.inner, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VertexAttribute {
    Vec3,
    Vec4,
}

impl VertexAttribute {
    fn vk(&self) -> vk::Format {
        match self {
            VertexAttribute::Vec3 => vk::Format::R32G32B32_SFLOAT,
            VertexAttribute::Vec4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }

    fn size(&self) -> u32 {
        match self {
            VertexAttribute::Vec3 => 3 * 4,
            VertexAttribute::Vec4 => 4 * 4,
        }
    }
}

pub struct VertexStageInfo<'a> {
    pub shader: &'a Shader,
    pub entrypoint: &'a str,
    pub attribs: Vec<VertexAttribute>,
    pub stride: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FragmentOutputInfo {
    pub format: Format,
    pub clear: bool,
    pub store: bool,
}

impl FragmentOutputInfo {
    fn load_op(&self) -> vk::AttachmentLoadOp {
        if self.clear {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        }
    }

    fn store_op(&self) -> vk::AttachmentStoreOp {
        if self.store {
            vk::AttachmentStoreOp::STORE
        } else {
            vk::AttachmentStoreOp::DONT_CARE
        }
    }
}

pub struct FragmentStageInfo<'a> {
    pub shader: &'a Shader,
    pub entrypoint: &'a str,
    pub outputs: Vec<FragmentOutputInfo>,
}

#[derive(Debug, Clone, Copy)]
pub enum RasterMode {
    Fill(f32),
    Outline(f32),
}

impl RasterMode {
    fn vk(&self) -> vk::PolygonMode {
        match self {
            RasterMode::Fill(_) => vk::PolygonMode::FILL,
            RasterMode::Outline(_) => vk::PolygonMode::LINE,
        }
    }

    fn width(&self) -> f32 {
        match self {
            RasterMode::Fill(w) => *w,
            RasterMode::Outline(w) => *w,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ClearValue {
    Colour([f32; 4]),
    Depth(f32, u32),
}

impl ClearValue {
    fn vk(&self) -> vk::ClearValue {
        match self {
            ClearValue::Colour(rgba) => vk::ClearValue {
                color: vk::ClearColorValue { float32: *rgba },
            },
            ClearValue::Depth(d, s) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue::default().depth(*d).stencil(*s),
            },
        }
    }
}

pub struct RenderOutput {
    inner: vk::Framebuffer,
    images: Vec<ImageView>,
}

impl Drop for RenderOutput {
    fn drop(&mut self) {
        unsafe {
            self.images[0]
                .dropper
                .image
                .inner
                .device
                .device
                .destroy_framebuffer(self.inner, None);
        }
    }
}

pub struct RenderPipeline {
    render_pass: vk::RenderPass,
    _output_infos: Vec<FragmentOutputInfo>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    dallocs: Vec<DAlloc>,
    device: Arc<DeviceDropper>,
}

impl RenderPipeline {
    fn new(
        device: &Arc<DeviceDropper>,
        vs_info: VertexStageInfo,
        fs_info: FragmentStageInfo,
        raster_info: RasterMode,
        descriptors: Vec<Vec<DBindingType>>,
        pc_size: u32,
    ) -> Result<Self, RhiError> {
        let dallocs: Vec<_> = descriptors
            .iter()
            .map(|d| DAlloc::new(&device, d))
            .collect::<Result<_, _>>()?;
        let attachment_access = ImageAccess::Attachment(RWAccess::ReadWrite);
        let rp_attachments: Vec<_> = fs_info
            .outputs
            .iter()
            .map(|a| {
                vk::AttachmentDescription::default()
                    .initial_layout(attachment_access.layout(a.format))
                    .final_layout(attachment_access.layout(a.format))
                    .format(a.format.vk())
                    .load_op(a.load_op())
                    .store_op(a.store_op())
                    .samples(vk::SampleCountFlags::TYPE_1)
            })
            .collect();
        let attachment_refs: Vec<_> = fs_info
            .outputs
            .iter()
            .enumerate()
            .map(|(i, a)| {
                vk::AttachmentReference::default()
                    .attachment(i as _)
                    .layout(attachment_access.layout(a.format))
            })
            .collect();
        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_refs);
        let rp_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&rp_attachments)
            .subpasses(core::slice::from_ref(&subpass_desc));
        let render_pass = unsafe {
            device
                .device
                .create_render_pass(&rp_create_info, None)
                .map_err(RhiError::CreateRenderPassError)?
        };
        let set_layouts: Vec<_> = dallocs.iter().map(|d| d.dsl).collect();
        let pc_info = vk::PushConstantRange::default().offset(0).size(pc_size);
        let mut pl_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if pc_size > 0 {
            pl_create_info = pl_create_info.push_constant_ranges(core::slice::from_ref(&pc_info));
        }
        let pipeline_layout = unsafe {
            device
                .device
                .create_pipeline_layout(&pl_create_info, None)
                .map_err(RhiError::CreatePipelineLayoutError)?
        };
        let dyn_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
        let vp_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let vert_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(vs_info.stride as _);
        let mut vert_attribs = vec![];
        let mut offset = 0;
        for (i, vsa) in vs_info.attribs.iter().enumerate() {
            let va = vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(i as _)
                .offset(offset)
                .format(vsa.vk());
            vert_attribs.push(va);
            offset += vsa.size();
        }
        let vert_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vert_attribs)
            .vertex_binding_descriptions(core::slice::from_ref(&vert_binding));
        let inp_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vs_main_name = init_helpers::safe_str_to_cstring(vs_info.entrypoint.to_string());
        let fs_main_name = init_helpers::safe_str_to_cstring(fs_info.entrypoint.to_string());
        let pipeline_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_info.shader.inner)
                .name(&vs_main_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_info.shader.inner)
                .name(&fs_main_name),
        ];
        let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(raster_info.vk())
            .line_width(raster_info.width())
            .depth_bias_enable(false);
        let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attach_blend_state: Vec<_> = (0..rp_attachments.len())
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&attach_blend_state);
        let p_create_info = vk::GraphicsPipelineCreateInfo::default()
            .render_pass(render_pass)
            .dynamic_state(&dyn_info)
            .viewport_state(&vp_state)
            .layout(pipeline_layout)
            .stages(&pipeline_stages)
            .vertex_input_state(&vert_state)
            .input_assembly_state(&inp_assembly)
            .rasterization_state(&raster_state)
            .multisample_state(&msaa_state)
            .color_blend_state(&color_blending);

        let pipeline = unsafe {
            device
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[p_create_info], None)
                .map_err(|(_, e)| RhiError::CreatePipelineError(e))?
                .remove(0)
        };
        Ok(Self {
            render_pass,
            _output_infos: fs_info.outputs,
            pipeline,
            pipeline_layout,
            dallocs,
            device: device.clone(),
        })
    }

    pub fn new_set(&mut self, idx: usize) -> Result<DSet, RhiError> {
        self.dallocs[idx].new_set()
    }

    pub fn new_output(&mut self, images: Vec<&ImageView>) -> Result<RenderOutput, RhiError> {
        let image_views_vk: Vec<_> = images.iter().map(|iv| iv.dropper.inner).collect();
        let fb_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(self.render_pass)
            .width(images[0].dropper.image.width)
            .height(images[0].dropper.image.height)
            .layers(images[0].dropper.image.depth)
            .attachments(&image_views_vk);
        let framebuffer = unsafe {
            self.device
                .device
                .create_framebuffer(&fb_create_info, None)
                .map_err(RhiError::CreateFramebufferError)?
        };
        Ok(RenderOutput {
            inner: framebuffer,
            images: images.into_iter().cloned().collect(),
        })
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}
