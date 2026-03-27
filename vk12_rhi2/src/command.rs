use std::sync::{Arc, Mutex};

use ash::vk;
use log::warn;
use rhi2::{command::CommandErr, image::Format, sync::PipelineStage};

use crate::{
    buffer::Buffer,
    device::DeviceDropper,
    graphics_pipeline::GraphicsPipeline,
    image::{Image, ImageAccess, ImageView},
    shader::ShaderSet,
    sync::{TaskFuture, TlSemPool, rhi2_pipe_stage_to_vk},
};

pub struct CmdBuffer {
    pub recording: bool,
    pub handle: vk::CommandBuffer,
    pub pool: Arc<CmdBufferGen>,
}

impl CmdBuffer {
    pub fn new_batch(pool: &Arc<CmdBufferGen>, count: usize) -> Result<Vec<Self>, String> {
        let rem_count = count;
        let mut cbs = vec![];
        let mut pool_mut = match pool.pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        while let Some(cb) = pool_mut.pop() {
            cbs.push(cb);
        }
        let cbs = unsafe {
            pool.device_dropper
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(pool.handle)
                        .command_buffer_count(rem_count as _),
                )
                .map_err(|e| format!("vk cmd buffers allocation failed: {e}"))?
        };
        let cbs: Vec<_> = cbs
            .into_iter()
            .map(|c| Self {
                recording: false,
                handle: c,
                pool: pool.clone(),
            })
            .collect();
        Ok(cbs)
    }

    pub fn start_recording(&mut self) -> Result<(), String> {
        if self.recording {
            return Ok(());
        }
        unsafe {
            self.pool
                .device_dropper
                .device
                .begin_command_buffer(self.handle, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| format!("cmd buffer beginning failed: {e}"))?;
        }
        self.recording = true;
        Ok(())
    }

    pub fn stop_recording(&mut self) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        unsafe {
            self.pool
                .device_dropper
                .device
                .end_command_buffer(self.handle)
                .map_err(|e| format!("cmd buffer stopping failed: {e}"))?;
        }
        self.recording = false;
        Ok(())
    }

    pub(crate) fn submit(
        &mut self,
        deps: Vec<(TaskFuture, PipelineStage, PipelineStage)>,
        sub: &TaskFuture,
    ) -> Result<(), String> {
        self.stop_recording()?;
        let mut sems = vec![];
        let mut sem_counts = vec![];
        let mut on_stgs = vec![];
        let mut by_stgs = vec![];

        for (tf, on, by) in deps {
            sems.push(tf.tl_sem);
            sem_counts.push(tf.count);
            on_stgs.push(rhi2_pipe_stage_to_vk(&on));
            by_stgs.push(rhi2_pipe_stage_to_vk(&by));
        }

        let mut tl_sem_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&sem_counts)
            .signal_semaphore_values(core::slice::from_ref(&sub.count));
        unsafe {
            self.pool
                .device_dropper
                .device
                .queue_submit(
                    self.pool.device_dropper.gfx_queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[self.handle])
                        .wait_semaphores(&sems)
                        .signal_semaphores(core::slice::from_ref(&sub.tl_sem))
                        .wait_dst_stage_mask(&on_stgs)
                        .push_next(&mut tl_sem_info)],
                    vk::Fence::null(),
                )
                .map_err(|e| format!("vk queue submit failed: {e}"))?;
        }
        Ok(())
    }
}

impl Drop for CmdBuffer {
    fn drop(&mut self) {
        self.stop_recording().inspect_err(|e| warn!("{e}")).ok();
    }
}

pub struct CmdBufferGen {
    pub pool: Arc<Mutex<Vec<CmdBuffer>>>,
    pub handle: vk::CommandPool,
    pub device_dropper: Arc<DeviceDropper>,
}

impl CmdBufferGen {
    pub fn new(device_dropper: &Arc<DeviceDropper>) -> Result<Self, String> {
        let handle = unsafe {
            device_dropper
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(device_dropper.gpu_info.gfx_qf as _)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("vk cmd pool creation failed: {e}"))?
        };
        Ok(Self {
            handle,
            pool: Arc::new(Mutex::new(vec![])),
            device_dropper: device_dropper.clone(),
        })
    }
}

impl Drop for CmdBufferGen {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .device
                .destroy_command_pool(self.handle, None);
        }
    }
}

pub struct GraphicsCommandRecorder {
    pub recorder: CommandRecorder,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl rhi2::command::GraphicsCommandRecorder for GraphicsCommandRecorder {
    type B = Buffer;

    type I = Image;

    type IV = ImageView;

    type SS = ShaderSet;

    fn bind_vbs(&mut self, vbs: &[&Self::B]) {
        let buffers: Vec<_> = vbs.iter().map(|b| b.handle).collect();
        unsafe {
            self.recorder.get_device().device.cmd_bind_vertex_buffers(
                self.recorder.inner.handle,
                0,
                &buffers,
                &vec![0; buffers.len()],
            );
        }
    }

    fn bind_ib(&mut self, ib: &Self::B, is_16bit: bool) {
        let index_type = if is_16bit {
            vk::IndexType::UINT16
        } else {
            vk::IndexType::UINT32
        };
        unsafe {
            self.recorder.get_device().device.cmd_bind_index_buffer(
                self.recorder.inner.handle,
                ib.handle,
                0,
                index_type,
            );
        }
    }

    fn bind_sets(&mut self, sets: &[&Self::SS]) {
        let dsets: Vec<_> = sets.iter().map(|s| s.handle).collect();
        unsafe {
            self.recorder.get_device().device.cmd_bind_descriptor_sets(
                self.recorder.inner.handle,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &dsets,
                &vec![0; dsets.len()],
            );
        }
    }

    fn set_pc(&mut self, data: &[u8]) {
        unsafe {
            self.recorder.get_device().device.cmd_push_constants(
                self.recorder.inner.handle,
                self.pipeline_layout,
                vk::ShaderStageFlags::ALL,
                0,
                data,
            );
        }
    }

    fn draw(&mut self, count: usize) {
        unsafe {
            self.recorder.get_device().device.cmd_draw(
                self.recorder.inner.handle,
                count as _,
                1,
                0,
                0,
            );
        }
    }

    fn draw_indexed(&mut self, vert_offset: usize, indx_offset: usize, count: usize) {
        unsafe {
            self.recorder.get_device().device.cmd_draw_indexed(
                self.recorder.inner.handle,
                count as _,
                1,
                indx_offset as _,
                vert_offset as _,
                0,
            );
        }
    }
}

pub struct CommandRecorder {
    pub inner: CmdBuffer,
    // pub image_layouts: HashMap<vk::Image, ImageAccess>,
    pub keep_alive_buffers: Vec<rhi2::Capped<Buffer>>,
    pub sync_pool: Arc<TlSemPool>,
    pub swapchain_img: Option<(vk::Image, Format)>,
}

impl CommandRecorder {
    pub fn new(
        cb_gen: &Arc<CmdBufferGen>,
        sync_pool: &Arc<TlSemPool>,
        count: usize,
    ) -> Result<Vec<Self>, String> {
        let cbs = CmdBuffer::new_batch(&cb_gen, count)
            .map_err(|e| format!("getting cmd buffers failed: {e}"))?;
        let mut crs: Vec<_> = cbs
            .into_iter()
            .map(|c| Self {
                inner: c,
                // image_layouts: HashMap::new(),
                keep_alive_buffers: vec![],
                sync_pool: sync_pool.clone(),
                swapchain_img: None,
            })
            .collect();
        for cr in &mut crs {
            cr.inner.start_recording()?;
        }
        Ok(crs)
    }

    fn get_device(&self) -> &DeviceDropper {
        &self.inner.pool.device_dropper
    }

    pub(crate) fn image_layout_update(&mut self, img: &Image, access: ImageAccess) {
        let ex_access = img.get_last_access();
        let aspect_mask = if img.format.is_depth() {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        if ex_access.layout == access.layout {
            return;
        }
        unsafe {
            self.get_device().device.cmd_pipeline_barrier(
                self.inner.handle,
                ex_access.psf,
                access.psf,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(img.handle)
                    .old_layout(ex_access.layout)
                    .new_layout(access.layout)
                    .src_access_mask(ex_access.access)
                    .dst_access_mask(access.access)
                    .src_queue_family_index(self.get_device().gpu_info.gfx_qf as _)
                    .dst_queue_family_index(self.get_device().gpu_info.gfx_qf as _)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect_mask)
                            .level_count(1)
                            .layer_count(img.layers),
                    )],
            );
        }
        img.set_last_access(access);
    }
}

impl rhi2::command::CommandRecorder for CommandRecorder {
    type B = Buffer;

    type I = Image;

    type IV = ImageView;

    type GP = GraphicsPipeline;

    type SS = ShaderSet;

    type GCR = GraphicsCommandRecorder;

    type TF = TaskFuture;

    fn copy_b2b(
        &mut self,
        src: &Self::B,
        src_offset: usize,
        dst: &Self::B,
        dst_offset: usize,
        copy_size: usize,
    ) {
        unsafe {
            self.get_device().device.cmd_copy_buffer(
                self.inner.handle,
                src.handle,
                dst.handle,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset as _)
                    .dst_offset(dst_offset as _)
                    .size(copy_size as _)],
            );
        }
    }

    fn copy_b2i(&mut self, src: &Self::B, dst: &Self::I) {
        let image_res = dst.res;
        let aspect_mask = if dst.format.is_depth() {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        self.image_layout_update(
            dst,
            ImageAccess {
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                access: vk::AccessFlags::TRANSFER_WRITE,
                psf: vk::PipelineStageFlags::TRANSFER,
            },
        );
        unsafe {
            self.get_device().device.cmd_copy_buffer_to_image(
                self.inner.handle,
                src.handle,
                dst.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_extent(
                        vk::Extent3D::default()
                            .width(image_res.0)
                            .height(image_res.1)
                            .depth(image_res.2),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(aspect_mask)
                            .layer_count(dst.layers)
                            .mip_level(0),
                    )],
            );
        }
        if self.swapchain_img.is_none() && dst.is_swapchain() {
            self.swapchain_img = Some((dst.handle, dst.format))
        }
    }

    fn graphics(
        mut self,
        pipeline: &mut Self::GP,
        color_ivs: Vec<&Self::IV>,
        color_clears: Vec<[f32; 4]>,
        depth_iv: Option<&Self::IV>,
        depth_clear: Option<f32>,
    ) -> Result<Self::GCR, (CommandErr, Self)> {
        let mut clear_values: Vec<_> = (0..color_ivs.len())
            .map(|_| vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            })
            .collect();
        for i in 0..color_clears.len() {
            clear_values[i].color.float32 = color_clears[i];
        }
        if let Some(_) = &depth_iv {
            let mut clr_val = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            };
            if let Some(clr) = depth_clear {
                clr_val.depth_stencil.depth = clr;
            }
            clear_values.push(clr_val);
        }
        let framebuffer = match pipeline
            .create_framebuffer(&color_ivs, depth_iv)
            .map_err(CommandErr::GcrCreate)
        {
            Ok(fb) => fb,
            Err(e) => return Err((e, self)),
        };
        let res = match color_ivs
            .first()
            .ok_or("no color attachments found".to_string())
            .map_err(CommandErr::GcrCreate)
        {
            Ok(iv) => iv,
            Err(e) => return Err((e, self)),
        }
        .image_holder
        .as_ref()
        .res;

        for a in &color_ivs {
            self.image_layout_update(
                a.image_holder.as_ref(),
                ImageAccess {
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    psf: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                },
            );
        }
        if let Some(a) = depth_iv.as_ref() {
            self.image_layout_update(
                a.image_holder.as_ref(),
                ImageAccess {
                    layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    psf: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                },
            );
        }

        unsafe {
            self.get_device().device.cmd_begin_render_pass(
                self.inner.handle,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pipeline.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D::default())
                            .extent(vk::Extent2D::default().width(res.0).height(res.1)),
                    )
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );
            self.get_device().device.cmd_bind_pipeline(
                self.inner.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
            self.get_device().device.cmd_set_viewport(
                self.inner.handle,
                0,
                &[vk::Viewport::default()
                    .width(res.0 as _)
                    .height(res.1 as _)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            self.get_device().device.cmd_set_scissor(
                self.inner.handle,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(vk::Extent2D::default().width(res.0).height(res.1))],
            );
        }
        if self.swapchain_img.is_none() {
            for a in color_ivs {
                if a.is_swapchain() {
                    let i = a.image_holder.as_ref();
                    self.swapchain_img = Some((i.handle, i.format));
                    break;
                }
            }
        }
        if self.swapchain_img.is_none() {
            if let Some(a) = depth_iv.as_ref() {
                if a.is_swapchain() {
                    let i = a.image_holder.as_ref();
                    self.swapchain_img = Some((i.handle, i.format));
                }
            }
        }
        Ok(GraphicsCommandRecorder {
            recorder: self,
            pipeline: pipeline.handle,
            pipeline_layout: pipeline.layout_handle,
        })
    }

    fn finish_graphics(gcr: Self::GCR) -> Self {
        unsafe {
            gcr.recorder
                .get_device()
                .device
                .cmd_end_render_pass(gcr.recorder.inner.handle);
        }
        gcr.recorder
    }

    fn blit(&mut self, src: &Self::I, dst: &Self::I) {
        self.image_layout_update(
            src,
            ImageAccess {
                layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                access: vk::AccessFlags::TRANSFER_READ,
                psf: vk::PipelineStageFlags::TRANSFER,
            },
        );
        self.image_layout_update(
            dst,
            ImageAccess {
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                access: vk::AccessFlags::TRANSFER_WRITE,
                psf: vk::PipelineStageFlags::TRANSFER,
            },
        );
        let src_res = src.res;
        let dst_res = dst.res;
        let src_aspect_mask = if src.format.is_depth() {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let dst_aspect_mask = if dst.format.is_depth() {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        unsafe {
            self.get_device().device.cmd_blit_image(
                self.inner.handle,
                src.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(src_res.0 as _)
                            .y(src_res.1 as _)
                            .z(src_res.2 as _),
                    ])
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D::default()
                            .x(dst_res.0 as _)
                            .y(dst_res.1 as _)
                            .z(dst_res.2 as _),
                    ])
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(src_aspect_mask)
                            .layer_count(src.layers)
                            .mip_level(0),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(dst_aspect_mask)
                            .layer_count(dst.layers)
                            .mip_level(0),
                    )],
                vk::Filter::NEAREST,
            );
        }
        if self.swapchain_img.is_none() {
            if src.is_swapchain() {
                self.swapchain_img = Some((src.handle, src.format))
            } else if dst.is_swapchain() {
                self.swapchain_img = Some((dst.handle, dst.format))
            }
        }
    }

    fn keep_buffer_alive(&mut self, buffer: rhi2::Capped<Self::B>) {
        self.keep_alive_buffers.push(buffer);
    }

    fn run(
        mut self,
        deps: Vec<(
            Self::TF,
            rhi2::sync::PipelineStage,
            rhi2::sync::PipelineStage,
        )>,
    ) -> Result<Self::TF, CommandErr> {
        let mut out_tf = TaskFuture::new(&self.sync_pool)
            .map_err(|e| format!("creating task future failed: {e}"))
            .map_err(CommandErr::RunErr)?;
        out_tf.increase_count(1);
        self.inner
            .submit(deps, &out_tf)
            .map_err(|e| format!("submitting command buffer failed: {e}"))
            .map_err(CommandErr::RunErr)?;
        out_tf.preserve_cmds.push(self.inner);
        out_tf.preserve_buffers.extend(self.keep_alive_buffers);
        Ok(out_tf)
    }
}
