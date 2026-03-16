use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    buffer::Buffer,
    device::DeviceDropper,
    graphics_pipeline::{GraphicsAttach, GraphicsPipeline},
    image::{Image, ImageAccess},
    shader::ShaderSet,
    sync::{
        BinSem, CpuFuture, CpuWaitable, Fence, GpuFuture, SyncPool, TlSem, rhi2_pipe_stage_to_vk,
    },
};

pub struct CmdPool {
    pub handle: vk::CommandPool,
    pub device_dropper: Arc<DeviceDropper>,
}

impl Drop for CmdPool {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .device
                .destroy_command_pool(self.handle, None);
        }
    }
}

pub struct CmdBuffer {
    pub handle: vk::CommandBuffer,
    pub cmd_pool: Arc<CmdPool>,
}

impl CmdBuffer {
    pub fn new_batch(
        device_dropper: &Arc<DeviceDropper>,
        count: usize,
    ) -> Result<Vec<Self>, String> {
        let pool = unsafe {
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
        let pool = Arc::new(CmdPool {
            handle: pool,
            device_dropper: device_dropper.clone(),
        });
        let cbs = unsafe {
            device_dropper
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(pool.handle)
                        .command_buffer_count(count as _),
                )
                .map_err(|e| format!("vk cmd buffers allocation failed: {e}"))?
        };
        let cbs: Vec<_> = cbs
            .into_iter()
            .map(|c| Self {
                handle: c,
                cmd_pool: pool.clone(),
            })
            .collect();
        Ok(cbs)
    }
}

pub struct CmdBufferGen {
    pool: Arc<Mutex<Vec<CmdBuffer>>>,
    device_dropper: Arc<DeviceDropper>,
}

impl CmdBufferGen {
    pub fn new(device_dropper: &Arc<DeviceDropper>) -> Self {
        Self {
            pool: Arc::new(Mutex::new(vec![])),
            device_dropper: device_dropper.clone(),
        }
    }

    pub fn get_cmd_buffers(&self, count: usize) -> Result<Vec<CmdBuffer>, String> {
        let mut rem_count = count;
        let mut cbs = vec![];
        let mut pool_mut = match self.pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        while let Some(cb) = pool_mut.pop() {
            cbs.push(cb);
            rem_count -= 1;
        }
        let new_cbs = CmdBuffer::new_batch(&self.device_dropper, rem_count)
            .map_err(|e| format!("new cmd buffer creation failed: {e}"))?;
        cbs.extend(new_cbs);
        Ok(cbs)
    }
}

pub struct GraphicsCommandRecorder {
    recorder: CommandRecorder,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl rhi2::command::GraphicsCommandRecorder for GraphicsCommandRecorder {
    type B = Buffer;

    type SS = ShaderSet;

    fn bind_vbs(&mut self, vbs: &[Self::B]) {
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

    fn bind_sets(&mut self, sets: &[Self::SS]) {
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
}

pub struct CommandRecorder {
    inner: CmdBuffer,
    image_layouts: HashMap<vk::Image, (ImageAccess, Arc<Mutex<ImageAccess>>)>,
    keep_alive_buffers: Vec<rhi2::Capped<Buffer>>,
    dependant_futures: Vec<(
        GpuFuture,
        rhi2::sync::PipelineStage,
        rhi2::sync::PipelineStage,
    )>,
    sync_pool: Arc<SyncPool>,
}

impl CommandRecorder {
    pub fn new(
        cb_gen: &CmdBufferGen,
        sync_pool: &Arc<SyncPool>,
        count: usize,
    ) -> Result<Vec<Self>, String> {
        let cbs = cb_gen
            .get_cmd_buffers(count)
            .map_err(|e| format!("getting cmd buffers failed: {e}"))?;
        let crs: Vec<_> = cbs
            .into_iter()
            .map(|c| Self {
                inner: c,
                image_layouts: HashMap::new(),
                keep_alive_buffers: vec![],
                dependant_futures: vec![],
                sync_pool: sync_pool.clone(),
            })
            .collect();
        Ok(crs)
    }

    fn get_device(&self) -> &DeviceDropper {
        &self.inner.cmd_pool.device_dropper
    }

    pub fn image_layout_update(&mut self, img: &Image, access: ImageAccess) {
        let ex_access = match self.image_layouts.get(&img.handle) {
            Some((ex_access, _)) => ex_access.clone(),
            None => {
                let ex_access = match img.last_access.lock() {
                    Ok(ll) => ll,
                    Err(e) => e.into_inner(),
                };
                ex_access.clone()
            }
        };
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
    }
}

impl rhi2::command::CommandRecorder for CommandRecorder {
    type B = Buffer;

    type I = Image;

    type GP = GraphicsPipeline;

    type GA = GraphicsAttach;

    type SS = ShaderSet;

    type GCR = GraphicsCommandRecorder;

    type GF = GpuFuture;

    type CF = CpuFuture;

    fn copy_b2b(&mut self, src: &Self::B, src_offset: usize, dst: &Self::B, dst_offset: usize) {
        unsafe {
            self.get_device().device.cmd_copy_buffer(
                self.inner.handle,
                src.handle,
                dst.handle,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset as _)
                    .dst_offset(dst_offset as _)
                    .size(vk::WHOLE_SIZE)],
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
    }

    fn graphics(self, pipeline: &Self::GP, attach: &Self::GA) -> Self::GCR {
        let mut clear_values: Vec<_> = attach
            .color_ivs
            .iter()
            .map(|_| vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            })
            .collect();
        if attach.depth_iv.is_some() {
            clear_values.push(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        }
        unsafe {
            self.get_device().device.cmd_begin_render_pass(
                self.inner.handle,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pipeline.render_pass)
                    .framebuffer(attach.framebuffer)
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D::default())
                            .extent(
                                vk::Extent2D::default()
                                    .width(attach.res.0)
                                    .height(attach.res.1),
                            ),
                    )
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );
            self.get_device().device.cmd_bind_pipeline(
                self.inner.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
        }
        GraphicsCommandRecorder {
            recorder: self,
            pipeline: pipeline.handle,
            pipeline_layout: pipeline.layout_handle,
        }
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
    }

    fn buffer_keep_alive(&mut self, buffer: rhi2::Capped<Self::B>) {
        self.keep_alive_buffers.push(buffer);
    }

    fn add_dependant(
        &mut self,
        cr: Self::GF,
        on_stage: rhi2::sync::PipelineStage,
        by_stage: rhi2::sync::PipelineStage,
    ) {
        self.dependant_futures.push((cr, on_stage, by_stage));
    }

    fn run_cpu_fut(mut self) -> Result<Self::CF, rhi2::command::CommandErr> {
        let force_legacy = false;
        let mut c_fut = if force_legacy {
            let tl_sem = TlSem::get(&self.sync_pool, 1)
                .map_err(|e| format!("failed getting TL semaphore: {e}"))
                .map_err(rhi2::command::CommandErr::FutureCreateErr)?
                .remove(0);
            CpuFuture::from_tl_sem(tl_sem, vec![])
        } else {
            let fence = Fence::get(&self.sync_pool, 1)
                .map_err(|e| format!("failed getting fence: {e}"))
                .map_err(rhi2::command::CommandErr::FutureCreateErr)?
                .remove(0);
            CpuFuture::from_fence(fence, vec![])
        };
        let mut wait_sems = vec![];
        let mut wait_nums = vec![];
        let mut on_stages = vec![];
        let mut by_stages = vec![];
        for (f, on, by) in self.dependant_futures.drain(..) {
            let (s, c) = f.get_wait_info();
            wait_sems.push(s);
            wait_nums.push(c);
            on_stages.push(rhi2_pipe_stage_to_vk(&on));
            by_stages.push(rhi2_pipe_stage_to_vk(&by));
            self.keep_alive_buffers.extend(f.preserve_buffers);
        }
        let fence = match &c_fut.inner {
            CpuWaitable::Fence(fence) => fence.handle,
            CpuWaitable::TlSem(_) => vk::Fence::null(),
        };
        let mut tl_submit_info =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_nums);
        match &c_fut.inner {
            CpuWaitable::Fence(_) => {}
            CpuWaitable::TlSem(tl_sem) => {
                tl_submit_info =
                    tl_submit_info.signal_semaphore_values(core::slice::from_ref(&tl_sem.count))
            }
        };
        unsafe {
            let mut submit_info = vk::SubmitInfo::default()
                .command_buffers(core::slice::from_ref(&self.inner.handle))
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&on_stages)
                .push_next(&mut tl_submit_info);
            match &c_fut.inner {
                CpuWaitable::Fence(_) => {}
                CpuWaitable::TlSem(tl_sem) => {
                    submit_info =
                        submit_info.signal_semaphores(core::slice::from_ref(&tl_sem.handle));
                }
            };
            self.get_device()
                .device
                .queue_submit(self.get_device().gfx_queue, &[submit_info], fence)
                .map_err(|e| format!("queue submission error: {e}"))
                .map_err(rhi2::command::CommandErr::FutureCreateErr)?;
        }
        c_fut.preserve_buffers = self.keep_alive_buffers;
        Ok(c_fut)
    }

    fn run_gpu_fut(mut self) -> Result<Self::GF, rhi2::command::CommandErr> {
        let force_legacy = false;
        let mut wait_sems = vec![];
        let mut wait_nums = vec![];
        let mut on_stages = vec![];
        let mut by_stages = vec![];
        for (f, on, by) in self.dependant_futures.drain(..) {
            let (s, c) = f.get_wait_info();
            wait_sems.push(s);
            wait_nums.push(c);
            on_stages.push(rhi2_pipe_stage_to_vk(&on));
            by_stages.push(rhi2_pipe_stage_to_vk(&by));
            self.keep_alive_buffers.extend(f.preserve_buffers);
        }
        let mut gfut = if force_legacy {
            let bin_sem = BinSem::get(&self.sync_pool, 1)
                .map_err(|e| format!("failed getting Bin semaphore: {e}"))
                .map_err(rhi2::command::CommandErr::FutureCreateErr)?
                .remove(0);
            GpuFuture::from_bin(bin_sem, vec![])
        } else {
            let tl_sem = TlSem::get(&self.sync_pool, 1)
                .map_err(|e| format!("failed getting TL semaphore: {e}"))
                .map_err(rhi2::command::CommandErr::FutureCreateErr)?
                .remove(0);
            GpuFuture::from_tl(tl_sem, vec![])
        };
        let (sig_sem, sig_count) = gfut.get_wait_info();
        let mut tl_submit_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_nums)
            .signal_semaphore_values(core::slice::from_ref(&sig_count));
        unsafe {
            self.get_device()
                .device
                .queue_submit(
                    self.get_device().gfx_queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&[self.inner.handle])
                        .wait_semaphores(&wait_sems)
                        .wait_dst_stage_mask(&on_stages)
                        .signal_semaphores(core::slice::from_ref(&sig_sem))
                        .push_next(&mut tl_submit_info)],
                    vk::Fence::null(),
                )
                .map_err(|e| format!("queue submission error: {e}"))
                .map_err(rhi2::command::CommandErr::FutureCreateErr)?;
        }
        gfut.preserve_buffers = self.keep_alive_buffers;
        Ok(gfut)
    }
}
