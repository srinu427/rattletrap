use std::sync::{Arc, Mutex};

use ash::vk;
use log::warn;
use rhi2::sync::TaskFuture as _;

use crate::{buffer::Buffer, command::CmdBuffer, device::DeviceDropper};

pub struct TlSemPool {
    pub tl_sem_pool: Arc<Mutex<Vec<(vk::Semaphore, u64)>>>,
    pub device_dropper: Arc<DeviceDropper>,
}

impl TlSemPool {
    pub fn new(device_dropper: &Arc<DeviceDropper>) -> Self {
        Self {
            tl_sem_pool: Arc::new(Mutex::new(vec![])),
            device_dropper: device_dropper.clone(),
        }
    }
}

impl Drop for TlSemPool {
    fn drop(&mut self) {
        unsafe {
            let mut sem_pool = match self.tl_sem_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            for (sem, _) in sem_pool.drain(..) {
                self.device_dropper.device.destroy_semaphore(sem, None);
            }
        }
    }
}

pub struct TaskFuture {
    pub preserve_futs: Vec<Self>,
    pub preserve_buffers: Vec<rhi2::Capped<Buffer>>,
    pub preserve_cmds: Vec<CmdBuffer>,
    pub tl_sem: vk::Semaphore,
    pub count: u64,
    pub sync_pool: Arc<TlSemPool>,
    waited: bool,
}

impl TaskFuture {
    pub fn new(sync_pool: &Arc<TlSemPool>) -> Result<Self, String> {
        let (tl_sem, count) = {
            let mut pool_mut = match sync_pool.tl_sem_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            match pool_mut.pop() {
                Some(s) => s,
                None => {
                    let mut tl_sem_info = vk::SemaphoreTypeCreateInfo::default()
                        .semaphore_type(vk::SemaphoreType::TIMELINE)
                        .initial_value(0);
                    let sem = unsafe {
                        sync_pool
                            .device_dropper
                            .device
                            .create_semaphore(
                                &vk::SemaphoreCreateInfo::default().push_next(&mut tl_sem_info),
                                None,
                            )
                            .map_err(|e| format!("creating vk tl sem failed: {e}"))?
                    };
                    (sem, 0)
                }
            }
        };
        Ok(Self {
            preserve_futs: vec![],
            preserve_buffers: vec![],
            preserve_cmds: vec![],
            tl_sem,
            count,
            sync_pool: sync_pool.clone(),
            waited: false,
        })
    }

    pub fn increase_count(&mut self, count: u64) {
        self.count += count;
    }

    pub fn sem_info(&self) -> (vk::Semaphore, u64) {
        (self.tl_sem, self.count)
    }
}

impl rhi2::sync::TaskFuture for TaskFuture {
    fn wait(&mut self) -> Result<(), rhi2::sync::SyncErr> {
        if self.waited {
            return Ok(());
        }
        unsafe {
            self.sync_pool
                .device_dropper
                .device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.tl_sem])
                        .values(&[self.count]),
                    u64::MAX,
                )
                .map_err(|e| format!("failed waiting for TL Semaphore: {e}"))
                .map_err(rhi2::sync::SyncErr::WaitErr)?;
        }
        self.waited = true;
        Ok(())
    }
}

impl Drop for TaskFuture {
    fn drop(&mut self) {
        self.wait().inspect_err(|e| warn!("{e}")).ok();
        let mut pool_mut = match self.sync_pool.tl_sem_pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        pool_mut.push((self.tl_sem, self.count));
    }
}

pub fn rhi2_pipe_stage_to_vk(ps: &rhi2::sync::PipelineStage) -> vk::PipelineStageFlags {
    match ps {
        rhi2::sync::PipelineStage::Top => vk::PipelineStageFlags::TOP_OF_PIPE,
        rhi2::sync::PipelineStage::Vertex => vk::PipelineStageFlags::VERTEX_SHADER,
        rhi2::sync::PipelineStage::DepthTest => vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        rhi2::sync::PipelineStage::Fragment => vk::PipelineStageFlags::FRAGMENT_SHADER,
        rhi2::sync::PipelineStage::AttachWrite => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        rhi2::sync::PipelineStage::Transfer => vk::PipelineStageFlags::TRANSFER,
        rhi2::sync::PipelineStage::Bottom => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    }
}
