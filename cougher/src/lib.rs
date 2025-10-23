use crate::traits::{
    ApiLoader, BufferUsage, CommandBuffer, CpuFuture, GpuContext, ImageFormat, ImageUsage,
    QueueType, Resolution2d, Swapchain,
};

pub mod backends;
pub mod traits;

pub struct Renderer<T: GpuContext> {
    ctx: T,
    swapchain: T::SwapchainType,
    bg_image: T::Image2dType,
    allocator: T::AllocatorType,
    command_buffers: Vec<T::CommandBufferType>,
    cpu_futures: Vec<T::CFutType>,
    gpu_futures: Vec<T::GFutType>,
    image_acquire_cfut: T::CFutType,
}

impl<T: GpuContext> Renderer<T> {
    pub fn from(ctx: T) -> Result<Self, T::E> {
        let swapchain = ctx.new_swapchain(ImageUsage::CopyDst.into())?;
        println!("count: {}", swapchain.images().len());
        let mut allocator = ctx.new_allocator()?;
        let command_buffers: Vec<_> = (0..swapchain.images().len())
            .map(|_| ctx.new_command_buffer(QueueType::Graphics))
            .collect::<Result<_, _>>()?;
        let cpu_futures: Vec<_> = (0..swapchain.images().len())
            .map(|_| ctx.new_cpu_future(false))
            .collect::<Result<_, _>>()?;
        let gpu_futures: Vec<_> = (0..swapchain.images().len())
            .map(|_| ctx.new_gpu_future())
            .collect::<Result<_, _>>()?;
        let bg_image_obj = image::open("default.png")?;
        let bg_image_data = bg_image_obj.as_bytes();
        println!("bg image len: {}", bg_image_data.len());
        let upload_cfut = ctx.new_cpu_future(false)?;
        let stage_buffer = ctx.new_buffer(
            &mut allocator,
            false,
            bg_image_data.len() as _,
            "stage_buffer",
            BufferUsage::TransferSrc.into(),
        )?;
        let bg_image = ctx.new_image_2d(
            &mut allocator,
            true,
            "bg_image",
            Resolution2d {
                width: bg_image_obj.width(),
                height: bg_image_obj.height(),
            },
            ImageFormat::Rgba8,
            ImageUsage::CopyDst | ImageUsage::CopySrc,
        )?;
        let mut stage_cmd_buffer = ctx.new_command_buffer(QueueType::Graphics)?;
        stage_cmd_buffer.add_image_2d_optimize_cmd(&bg_image, ImageUsage::None);
        stage_cmd_buffer.copy_buffer_to_image_2d_cmd(&stage_buffer, &bg_image);
        stage_cmd_buffer.add_image_2d_optimize_cmd(&bg_image, ImageUsage::CopySrc);
        stage_cmd_buffer.build()?;
        stage_cmd_buffer.emit_cpu_future_on_finish(&upload_cfut);
        stage_cmd_buffer.submit()?;
        upload_cfut.wait()?;
        drop(stage_buffer);
        let image_acquire_cfut = ctx.new_cpu_future(false)?;

        Ok(Self {
            ctx,
            swapchain,
            bg_image,
            allocator,
            command_buffers,
            cpu_futures,
            gpu_futures,
            image_acquire_cfut,
        })
    }

    pub fn draw(&mut self) -> Result<(), T::E> {
        let next_img = self
            .swapchain
            .get_next_image(Some(&self.image_acquire_cfut), None)?;
        self.image_acquire_cfut.wait()?;

        self.command_buffers[next_img as usize].reset()?;
        if !self.swapchain.is_optimized() {
            for img in self.swapchain.images() {
                self.command_buffers[next_img as usize]
                    .add_image_2d_optimize_cmd(img, ImageUsage::None);
                self.command_buffers[next_img as usize]
                    .add_image_2d_optimize_cmd(img, ImageUsage::Present);
            }
        } else {
            self.command_buffers[next_img as usize].add_image_2d_optimize_cmd(
                &self.swapchain.images()[next_img as usize],
                ImageUsage::Present,
            );
        }

        self.command_buffers[next_img as usize]
            .add_blit_image_2d_cmd(&self.bg_image, &self.swapchain.images()[next_img as usize]);
        self.command_buffers[next_img as usize].add_image_2d_optimize_cmd(
            &self.swapchain.images()[next_img as usize],
            ImageUsage::Present,
        );
        self.command_buffers[next_img as usize].build()?;
        self.command_buffers[next_img as usize]
            .emit_cpu_future_on_finish(&self.cpu_futures[next_img as usize]);
        self.command_buffers[next_img as usize].submit()?;
        self.cpu_futures[next_img as usize].wait()?;

        self.swapchain.present(next_img, &[])?;

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UnivRendererError {
    #[error("Vk12 Error: {0}")]
    Vk12Error(#[from] backends::vulkan_1_2::V12ContextError),
    #[error("Vk12 Load Error: {0}")]
    Vk12LoadError(#[from] backends::vulkan_1_2::V12ApiLoaderError),
}

pub enum UnivRenderer {
    Vk12(Renderer<backends::vulkan_1_2::V12Context>),
}

impl UnivRenderer {
    pub fn new(window: winit::window::Window) -> Result<Self, UnivRendererError> {
        let loader = backends::vulkan_1_2::V12ApiLoader::new(window)?;
        let gpu = loader.list_supported_gpus().remove(0);
        let ctx = loader.new_gpu_context(gpu)?;
        let renderer = Renderer::from(ctx)?;
        Ok(Self::Vk12(renderer))
    }

    pub fn draw(&mut self) -> Result<(), UnivRendererError> {
        match self {
            UnivRenderer::Vk12(renderer) => renderer.draw()?,
        };
        Ok(())
    }
}
