use std::{
    ops::Range,
    sync::{Arc, Mutex, PoisonError},
};

use anyhow::Context;
use ash::vk;

use crate::{
    device::DeviceDropper,
    pipeline::{DSet, GraphicsPipeline, GraphicsPipelineDropper},
    resource::{Buffer, BufferView, Image, ImageAccess, ImageView},
    sync::{Sem, WaitResult},
};

static COMMAND_BUFFER_ALLOC_BATCH_SIZE: usize = 8;

struct CmdPoolDropper {
    pool: Mutex<Vec<vk::CommandBuffer>>,
    handle: vk::CommandPool,
    dd: Arc<DeviceDropper>,
}

impl Drop for CmdPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.dd.device.destroy_command_pool(self.handle, None);
        }
    }
}

pub(crate) struct CmdPool {
    dropper: Arc<CmdPoolDropper>,
}

impl CmdPool {
    pub(crate) fn new(dd: &Arc<DeviceDropper>) -> anyhow::Result<Self> {
        let cmd_pool = unsafe {
            dd.device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(dd.gpu_info.gfx_qf as u32)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .context("command pool init failed")?
        };
        Ok(Self {
            dropper: Arc::new(CmdPoolDropper {
                pool: Mutex::new(vec![]),
                handle: cmd_pool,
                dd: dd.clone(),
            }),
        })
    }

    pub(crate) fn get_cb(&self) -> anyhow::Result<CmdBuf> {
        let ex_cb = self
            .dropper
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let handle = match ex_cb {
            Some(cb) => cb,
            None => {
                let new_cbs = unsafe {
                    self.dropper
                        .dd
                        .device
                        .allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .level(vk::CommandBufferLevel::PRIMARY)
                                .command_pool(self.dropper.handle)
                                .command_buffer_count(COMMAND_BUFFER_ALLOC_BATCH_SIZE as _),
                        )
                        .context("command buffer allocation failed")?
                };
                self.dropper
                    .pool
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend(&new_cbs[1..]);
                new_cbs[0]
            }
        };
        let mut cb = CmdBuf {
            recording: false,
            handle,
            dropper: self.dropper.clone(),
        };
        cb.begin()?;
        Ok(cb)
    }
}

pub(crate) struct CmdBuf {
    recording: bool,
    handle: vk::CommandBuffer,
    dropper: Arc<CmdPoolDropper>,
}

impl CmdBuf {
    fn begin(&mut self) -> anyhow::Result<()> {
        if self.recording {
            return Ok(());
        }
        unsafe {
            self.dropper
                .dd
                .device
                .begin_command_buffer(self.handle, &vk::CommandBufferBeginInfo::default())
                .context("command buffer begin failed")?
        }
        self.recording = true;
        Ok(())
    }

    fn end(&mut self) -> anyhow::Result<()> {
        if !self.recording {
            return Ok(());
        }
        unsafe {
            self.dropper
                .dd
                .device
                .end_command_buffer(self.handle)
                .context("command buffer end failed")?
        }
        self.recording = false;
        Ok(())
    }
}

impl Drop for CmdBuf {
    fn drop(&mut self) {
        if let Err(e) = self.end() {
            log::warn!("command buffer record ending failed while returning to pool: {e}");
        }
        self.dropper
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.handle);
    }
}

pub struct Task {
    pub(crate) cb: CmdBuf,
    pub(crate) sem: Sem,
    // store resources so they dont get deleted
    pub(crate) preserve_bufs: Vec<Buffer>,
    pub(crate) preserve_imgs: Vec<Image>,
    pub(crate) preserve_views: Vec<ImageView>,
    pub(crate) preserve_gps: Vec<Arc<GraphicsPipelineDropper>>,
    pub(crate) swapchain_image: Option<Image>,
}

impl Task {
    pub fn run(mut self) -> anyhow::Result<GpuFuture> {
        if let Some(sw_img) = self.swapchain_image.take() {
            self.update_image_accesses(
                &sw_img,
                ImageAccess {
                    layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    access_flags: vk::AccessFlags::MEMORY_READ,
                    access_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                },
            );
        }
        self.cb.end()?;
        self.sem.sem_val.val += 1;
        unsafe {
            self.cb
                .dropper
                .dd
                .device
                .queue_submit(
                    self.cb.dropper.dd.gfx_queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[self.cb.handle])
                        .signal_semaphores(&[self.sem.sem_val.handle])
                        .push_next(
                            &mut vk::TimelineSemaphoreSubmitInfo::default()
                                .signal_semaphore_values(&[self.sem.sem_val.val]),
                        )],
                    vk::Fence::null(),
                )
                .context("command buffer submission failed")?
        }
        Ok(GpuFuture { task: self })
    }

    pub fn copy_b2b(&mut self, src: BufferView, dst: BufferView) {
        let copy_size = (src.range.end - src.range.start).min(dst.range.end - dst.range.start);
        if copy_size == 0 {
            return;
        }
        unsafe {
            self.cb.dropper.dd.device.cmd_copy_buffer(
                self.cb.handle,
                src.buffer.dropper.handle,
                dst.buffer.dropper.handle,
                &[vk::BufferCopy::default()
                    .src_offset(src.range.start)
                    .dst_offset(dst.range.start)
                    .size(copy_size)],
            );
        }
        self.preserve_bufs.push(src.buffer.clone());
        self.preserve_bufs.push(dst.buffer.clone());
    }

    fn update_image_accesses(&mut self, image: &Image, new_access: ImageAccess) {
        let old_access = image
            .dropper
            .last_access
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        unsafe {
            self.cb.dropper.dd.device.cmd_pipeline_barrier(
                self.cb.handle,
                old_access.access_stage,
                new_access.access_stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image.dropper.handle)
                    .src_access_mask(old_access.access_flags)
                    .old_layout(old_access.layout)
                    .src_queue_family_index(self.cb.dropper.dd.gpu_info.gfx_qf as _)
                    .dst_access_mask(new_access.access_flags)
                    .new_layout(new_access.layout)
                    .dst_queue_family_index(self.cb.dropper.dd.gpu_info.gfx_qf as _)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(image.info.format.aspect_flag())
                            .layer_count(image.info.layers)
                            .level_count(image.info.mip_levels),
                    )],
            );
        }
    }

    fn check_and_store_sw_img(&mut self, image: &Image) {
        if image.dropper.mem.is_none() && self.swapchain_image.is_none() {
            self.swapchain_image.replace(image.clone());
        }
    }

    pub fn copy_b2i(
        &mut self,
        src: BufferView,
        dst: &Image,
        mip_level: u32,
        layer_range: Range<u32>,
    ) {
        self.update_image_accesses(
            dst,
            ImageAccess {
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                access_flags: vk::AccessFlags::TRANSFER_WRITE,
                access_stage: vk::PipelineStageFlags::TRANSFER,
            },
        );
        self.check_and_store_sw_img(dst);
        unsafe {
            self.cb.dropper.dd.device.cmd_copy_buffer_to_image(
                self.cb.handle,
                src.buffer.dropper.handle,
                dst.dropper.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(src.range.start)
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: dst.info.res.0,
                        height: dst.info.res.1,
                        depth: dst.info.res.2,
                    })
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(dst.info.format.aspect_flag())
                            .mip_level(mip_level)
                            .base_array_layer(layer_range.start)
                            .layer_count(layer_range.end - layer_range.start),
                    )],
            );
        }
        self.preserve_bufs.push(src.buffer.clone());
        self.preserve_imgs.push(dst.clone());
    }

    fn update_image_view_accesses(&self, view: &ImageView, new_access: ImageAccess) {
        let old_access = view
            .image
            .dropper
            .last_access
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        unsafe {
            self.cb.dropper.dd.device.cmd_pipeline_barrier(
                self.cb.handle,
                old_access.access_stage,
                new_access.access_stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(view.image.dropper.handle)
                    .src_access_mask(old_access.access_flags)
                    .old_layout(old_access.layout)
                    .src_queue_family_index(self.cb.dropper.dd.gpu_info.gfx_qf as _)
                    .dst_access_mask(new_access.access_flags)
                    .new_layout(new_access.layout)
                    .dst_queue_family_index(self.cb.dropper.dd.gpu_info.gfx_qf as _)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(view.image.info.format.aspect_flag())
                            .layer_count(view.image.info.layers)
                            .level_count(view.image.info.mip_levels),
                    )],
            );
        }
    }

    pub fn graphics<'a>(
        &'a mut self,
        pipeline: &'a mut GraphicsPipeline,
        attachments: Vec<ImageView>,
        clears: Vec<ClearValue>,
    ) -> anyhow::Result<GraphicsTask<'a>> {
        self.cb.begin()?;
        let fb = pipeline.get_fb(&attachments)?;
        if attachments.len() != clears.len() {
            return Err(anyhow::Error::msg(
                "attachment count and clear count is different",
            ));
        }
        for a in &attachments {
            let new_access = if a.image.info.format.is_depth() {
                ImageAccess {
                    layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    access_flags: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    access_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                }
            } else {
                ImageAccess {
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    access_flags: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    access_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                }
            };
            self.update_image_view_accesses(a, new_access);
            self.check_and_store_sw_img(&a.image);
        }
        let width = attachments[0].image.info.res.0;
        let height = attachments[0].image.info.res.1;
        unsafe {
            self.cb.dropper.dd.device.cmd_begin_render_pass(
                self.cb.handle,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pipeline.dropper.render_pass)
                    .framebuffer(fb)
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D::default())
                            .extent(
                                vk::Extent2D::default()
                                    .width(width as _)
                                    .height(height as _),
                            ),
                    )
                    .clear_values(&clears.iter().map(|c| c.to_vk()).collect::<Vec<_>>()),
                vk::SubpassContents::INLINE,
            );
            self.cb.dropper.dd.device.cmd_set_viewport(
                self.cb.handle,
                0,
                &[vk::Viewport::default()
                    .width(width as _)
                    .height(height as _)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            self.cb.dropper.dd.device.cmd_set_scissor(
                self.cb.handle,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(
                        vk::Extent2D::default()
                            .width(width as _)
                            .height(height as _),
                    )],
            );
            self.cb.dropper.dd.device.cmd_bind_pipeline(
                self.cb.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.dropper.pipeline,
            );
        }
        self.preserve_gps.push(pipeline.dropper.clone());
        Ok(GraphicsTask {
            task: self,
            pipeline,
            attachments,
        })
    }
}

#[derive(Debug, Clone)]
pub enum ClearValue {
    Color([f32; 4]),
    Depth(f32),
}

impl ClearValue {
    pub(crate) fn to_vk(&self) -> vk::ClearValue {
        match self {
            ClearValue::Color(vals) => vk::ClearValue {
                color: vk::ClearColorValue { float32: *vals },
            },
            ClearValue::Depth(val) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: *val,
                    stencil: 0,
                },
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum DrawInfo {
    Raw {
        offset: u32,
        count: u32,
    },
    Indexed {
        vb_offset: u32,
        ib_offset: u32,
        count: u32,
    },
}

pub struct GraphicsTask<'a> {
    task: &'a mut Task,
    pipeline: &'a mut GraphicsPipeline,
    attachments: Vec<ImageView>,
}

impl<'a> GraphicsTask<'a> {
    pub fn bind_vb(&mut self, vb: &Buffer) {
        unsafe {
            self.task.cb.dropper.dd.device.cmd_bind_vertex_buffers(
                self.task.cb.handle,
                0,
                &[vb.dropper.handle],
                &[0],
            );
        }
    }

    pub fn bind_ib(&mut self, ib: &Buffer, is_16bit: bool) {
        unsafe {
            self.task.cb.dropper.dd.device.cmd_bind_index_buffer(
                self.task.cb.handle,
                ib.dropper.handle,
                0,
                if is_16bit {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                },
            );
        }
    }

    pub fn bind_set(&mut self, set_idx: usize, set: &DSet) {
        unsafe {
            self.task.cb.dropper.dd.device.cmd_bind_descriptor_sets(
                self.task.cb.handle,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.dropper.pipeline_layout,
                set_idx as _,
                &[set.set],
                &[],
            );
        }
    }

    pub fn set_pc(&mut self, data: &[u8]) {
        unsafe {
            self.task.cb.dropper.dd.device.cmd_push_constants(
                self.task.cb.handle,
                self.pipeline.dropper.pipeline_layout,
                vk::ShaderStageFlags::ALL,
                0,
                data,
            );
        }
    }

    pub fn draw(&mut self, draws: Vec<DrawInfo>) {
        for draw in &draws {
            match draw {
                DrawInfo::Raw { offset, count } => unsafe {
                    self.task.cb.dropper.dd.device.cmd_draw(
                        self.task.cb.handle,
                        *count,
                        1,
                        *offset,
                        0,
                    );
                },
                DrawInfo::Indexed {
                    vb_offset,
                    ib_offset,
                    count,
                } => unsafe {
                    self.task.cb.dropper.dd.device.cmd_draw_indexed(
                        self.task.cb.handle,
                        *count,
                        1,
                        *ib_offset,
                        *vb_offset as _,
                        0,
                    );
                },
            }
        }
    }
}

impl<'a> Drop for GraphicsTask<'a> {
    fn drop(&mut self) {
        unsafe {
            self.task
                .cb
                .dropper
                .dd
                .device
                .cmd_end_render_pass(self.task.cb.handle);
        }
        self.task.preserve_views.extend(self.attachments.drain(..));
    }
}

pub struct GpuFuture {
    task: Task,
}

impl GpuFuture {
    pub fn wait(self) -> anyhow::Result<()> {
        match self.task.sem.wait(u64::MAX) {
            WaitResult::Success => Ok(()),
            WaitResult::Timeout => Err(anyhow::Error::msg("waiting forever didnt work!")),
            WaitResult::Error(e) => Err(anyhow::Error::msg(e)),
        }
    }
}

impl Future for GpuFuture {
    type Output = anyhow::Result<()>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let wait_res = self.task.sem.wait(0);
        match wait_res {
            WaitResult::Success => std::task::Poll::Ready(Ok(())),
            WaitResult::Timeout => std::task::Poll::Pending,
            WaitResult::Error(e) => std::task::Poll::Ready(Err(anyhow::Error::msg(e))),
        }
    }
}
