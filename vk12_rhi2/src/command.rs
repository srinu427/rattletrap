use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ash::vk;
use rhi2::command::CommandErr;

use crate::{
    buffer::Buffer,
    device::DeviceDropper,
    graphics_pipeline::GraphicsPipeline,
    image::{Image, ImageAccess, ImageView},
    shader::ShaderSet,
    sync::{SyncPool, TaskFuture, rhi2_pipe_stage_to_vk},
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
    pub pool: Arc<Mutex<Vec<CmdBuffer>>>,
    pub device_dropper: Arc<DeviceDropper>,
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
    pub recorder: CommandRecorder,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
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
    pub inner: CmdBuffer,
    pub image_layouts: HashMap<vk::Image, (ImageAccess, Arc<Mutex<ImageAccess>>)>,
    pub keep_alive_buffers: Vec<rhi2::Capped<Buffer>>,
    pub sync_pool: Arc<SyncPool>,
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

    type IV = ImageView;

    type GP = GraphicsPipeline;

    type SS = ShaderSet;

    type GCR = GraphicsCommandRecorder;

    type TF = TaskFuture;

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

    fn graphics(
        self,
        pipeline: &mut Self::GP,
        color_ivs: &[Self::IV],
        depth_iv: Option<&Self::IV>,
    ) -> Result<Self::GCR, (CommandErr, Self)> {
        let mut clear_values: Vec<_> = color_ivs
            .iter()
            .map(|_| vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            })
            .collect();
        if depth_iv.is_some() {
            clear_values.push(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        }
        let framebuffer = match pipeline
            .create_framebuffer(color_ivs, depth_iv)
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
    }

    fn keep_buffer_alive(&mut self, buffer: rhi2::Capped<Self::B>) {
        self.keep_alive_buffers.push(buffer);
    }

    fn run(
        self,
        deps: Vec<(
            Self::TF,
            rhi2::sync::PipelineStage,
            rhi2::sync::PipelineStage,
        )>,
    ) -> Result<Self::TF, rhi2::command::CommandErr> {
        let mut sems = vec![];
        let mut sem_counts = vec![];
        let mut on_stages = vec![];
        let mut by_stages = vec![];

        for (tf, on, by) in &deps {
            let sem_infos = tf.sem_infos();
            let on_stage = rhi2_pipe_stage_to_vk(on);
            let by_stage = rhi2_pipe_stage_to_vk(by);
            for (sem, count) in sem_infos {
                sems.push(sem);
                sem_counts.push(count);
                on_stages.push(on_stage);
                by_stages.push(by_stage);
            }
        }

        unsafe {}
        todo!()
    }
}
