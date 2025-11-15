use std::mem::ManuallyDrop;

use ash::vk;
use ash::vk::Handle;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocationError, MemoryLocation};

mod device;
mod image2d;
mod instance;

use crate::build_init_cleanup_struct;
use crate::vk12::device::Vk12Device;
use crate::vk12::image2d::Image2d;

build_init_cleanup_struct!(
    InitCommandPool,
    vk::CommandPool,
    self,
    self.device.destroy_command_pool(self.inner, None)
);

build_init_cleanup_struct!(
    InitFence,
    vk::Fence,
    self,
    self.device.destroy_fence(self.inner, None)
);

build_init_cleanup_struct!(
    InitImage,
    vk::Image,
    self,
    self.device.destroy_image(self.inner, None)
);

pub struct CompositeInput<'a> {
    pub image: &'a Image2d,
    pub range: [(f32, f32); 2],
}

fn composite_images(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    dst: vk::Image,
    dst_res: vk::Extent2D,
    inputs: Vec<CompositeInput>,
) {
    unsafe {
        for inp in inputs {
            device.cmd_blit_image(
                cmd_buffer,
                inp.image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(Image2d::image_subresource_layers(false, false))
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(inp.image.extent.width as _)
                            .y(inp.image.extent.height as _)
                            .z(1),
                    ])
                    .dst_subresource(Image2d::image_subresource_layers(false, false))
                    .dst_offsets([
                        vk::Offset3D::default()
                            .x((inp.range[0].0 * dst_res.width as f32) as _)
                            .y((inp.range[0].1 * dst_res.width as f32) as _),
                        vk::Offset3D::default()
                            .x((inp.range[0].0 * dst_res.width as f32) as _)
                            .y((inp.range[0].1 * dst_res.width as f32) as _)
                            .z(1),
                    ])],
                vk::Filter::NEAREST,
            );
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Vk12RendererError {
    #[error("Error creating Vulkan Memory Allocator: {0}")]
    AllocatorInitError(AllocationError),
    #[error("Error creating Vulkan Command Pool: {0}")]
    CommandPoolCreateError(vk::Result),
    #[error("Error creating Vulkan Fence: {0}")]
    FenceCreateError(vk::Result),
    #[error("Error allocating Vulkan Command Buffers: {0}")]
    CommandBufferAllocateError(vk::Result),
}

pub struct Vk12Renderer {
    swapchain_init_done: bool,
    draw_fences: Vec<vk::Fence>,
    draw_cmd_buffers: Vec<vk::CommandBuffer>,
    image_acquire_fence: vk::Fence,
    command_pool: vk::CommandPool,
    allocator: Allocator,
    device: Vk12Device,
}

impl Vk12Renderer {
    pub fn new(device: Vk12Device) -> Result<Self, Vk12RendererError> {
        let command_pool = InitCommandPool {
            drop: true,
            inner: unsafe {
                device
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(device.g_queue_fam),
                        None,
                    )
                    .map_err(Vk12RendererError::CommandPoolCreateError)?
            },
            device: &device.device,
        };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: device.instance.instance.clone(),
            device: device.device.clone(),
            physical_device: device.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(Vk12RendererError::AllocatorInitError)?;
        let image_acquire_fence = InitFence {
            drop: true,
            inner: unsafe {
                device
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(Vk12RendererError::FenceCreateError)?
            },
            device: &device.device,
        };
        let draw_cmd_buffers = unsafe {
            device
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(command_pool.read())
                        .command_buffer_count(device.swapchain_data.images.len() as _),
                )
                .map_err(Vk12RendererError::CommandBufferAllocateError)?
        };
        let draw_fences: Vec<_> = (0..device.swapchain_data.images.len())
            .map(|_| {
                let fence = InitFence {
                    drop: true,
                    inner: unsafe {
                        device
                            .device
                            .create_fence(&vk::FenceCreateInfo::default(), None)?
                    },
                    device: &device.device,
                };
                Ok(fence)
            })
            .collect::<Result<_, _>>()
            .map_err(Vk12RendererError::FenceCreateError)?;

        Ok(Self {
            swapchain_init_done: false,
            draw_fences: draw_fences.into_iter().map(|f| f.take()).collect(),
            draw_cmd_buffers,
            image_acquire_fence: image_acquire_fence.take(),
            command_pool: command_pool.take(),
            allocator,
            device,
        })
    }
}

impl Drop for Vk12Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
