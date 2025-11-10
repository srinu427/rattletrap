use crate::traits::{
    ApiLoader, Buffer, BufferUsage, CpuFuture, GpuCommand, GpuContext, GpuExecutor, ImageFormat,
    ImageUsage, Resolution2d, Swapchain,
};

pub mod backends;
pub mod render_objs;
pub mod traits;

pub struct Renderer<T: GpuContext> {
    ctx: T,
    swapchain: T::SwapchainType,
    bg_image: T::I2dType,
    allocator: T::MP,
    executor: T::QType,
    cpu_futures: Vec<T::FenType>,
    gpu_futures: Vec<T::SemType>,
    image_acquire_cfut: T::FenType,
}

impl<T: GpuContext> Renderer<T> {
    pub fn from(mut ctx: T) -> Result<Self, T::E> {
        let swapchain = ctx.new_swapchain(ImageUsage::CopyDst.into())?;
        let mut allocator = ctx.new_allocator()?;
        let mut executor = ctx.get_queue()?;
        for i in 0..swapchain.images().len() {
            executor.new_command_list(&format!("render_cmd_list_{i}"))?;
        }
        let cpu_futures: Vec<_> = (0..swapchain.images().len())
            .map(|_| ctx.new_cpu_future(true))
            .collect::<Result<_, _>>()?;
        let gpu_futures: Vec<_> = (0..swapchain.images().len())
            .map(|_| ctx.new_gpu_future())
            .collect::<Result<_, _>>()?;
        let bg_image_obj = image::open("default.png")?;
        let bg_image_data = bg_image_obj.as_bytes();
        let upload_cfut = ctx.new_cpu_future(false)?;
        let mut stage_buffer = ctx.new_buffer(
            &mut allocator,
            false,
            bg_image_data.len() as _,
            "stage_buffer",
            BufferUsage::TransferSrc.into(),
        )?;
        stage_buffer.write_data(0, &bg_image_data)?;
        let bg_image = ctx.new_image_2d(
            &mut allocator,
            true,
            "bg_image",
            Resolution2d {
                width: bg_image_obj.width(),
                height: bg_image_obj.height(),
            },
            ImageFormat::Rgba8Srgb,
            ImageUsage::CopyDst | ImageUsage::CopySrc,
        )?;
        executor.new_command_list("bg_image_copy")?;
        executor.update_command_list(
            "bg_image_copy",
            vec![
                GpuCommand::Image2dUsageHint {
                    image: &bg_image,
                    usage: ImageUsage::None,
                },
                GpuCommand::CopyBufferToImage2d {
                    src: &stage_buffer,
                    dst: &bg_image,
                },
                GpuCommand::Image2dUsageHint {
                    image: &bg_image,
                    usage: ImageUsage::CopySrc,
                },
            ],
        )?;
        executor.run_command_lists(&["bg_image_copy"], vec![], vec![], Some(&upload_cfut))?;
        upload_cfut.wait()?;
        drop(stage_buffer);
        let image_acquire_cfut = ctx.new_cpu_future(false)?;

        Ok(Self {
            ctx,
            swapchain,
            bg_image,
            allocator,
            executor,
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

        self.cpu_futures[next_img as usize].wait()?;
        let mut commands = vec![];
        if !self.swapchain.is_optimized() {
            for img in self.swapchain.images() {
                commands.push(GpuCommand::Image2dUsageHint {
                    image: img,
                    usage: ImageUsage::None,
                });
                commands.push(GpuCommand::Image2dUsageHint {
                    image: img,
                    usage: ImageUsage::Present,
                });
            }
        } else {
            commands.push(GpuCommand::Image2dUsageHint {
                image: &self.swapchain.images()[next_img as usize],
                usage: ImageUsage::Present,
            });
        }
        commands.push(GpuCommand::BlitImage2d {
            src: &self.bg_image,
            dst: &self.swapchain.images()[next_img as usize],
        });
        commands.push(GpuCommand::Image2dUsageHint {
            image: &self.swapchain.images()[next_img as usize],
            usage: ImageUsage::Present,
        });

        self.executor
            .update_command_list(&format!("render_cmd_list_{next_img}"), commands)?;
        self.executor.run_command_lists(
            &[&format!("render_cmd_list_{next_img}")],
            vec![],
            vec![&self.gpu_futures[next_img as usize]],
            Some(&self.cpu_futures[next_img as usize]),
        )?;

        self.swapchain
            .present(next_img, &[&self.gpu_futures[next_img as usize]])?;

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
