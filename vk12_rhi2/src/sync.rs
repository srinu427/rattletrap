use std::sync::{Arc, Mutex};

use ash::vk;

use crate::{buffer::Buffer, device::DeviceDropper};

pub struct SyncPool {
    pub tl_sem_pool: Arc<Mutex<Vec<(vk::Semaphore, u64)>>>,
    pub bin_sem_pool: Arc<Mutex<Vec<vk::Semaphore>>>,
    pub device_dropper: Arc<DeviceDropper>,
}

impl SyncPool {
    pub fn new(device_dropper: &Arc<DeviceDropper>) -> Self {
        Self {
            tl_sem_pool: Arc::new(Mutex::new(vec![])),
            bin_sem_pool: Arc::new(Mutex::new(vec![])),
            device_dropper: device_dropper.clone(),
        }
    }
}

impl Drop for SyncPool {
    fn drop(&mut self) {
        unsafe {
            let mut sem_pool = match self.tl_sem_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            for (sem, _) in sem_pool.drain(..) {
                self.device_dropper.device.destroy_semaphore(sem, None);
            }
            let mut sem_pool = match self.bin_sem_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            for sem in sem_pool.drain(..) {
                self.device_dropper.device.destroy_semaphore(sem, None);
            }
        }
    }
}

pub struct TaskFuture {
    pub preserve_buffers: Vec<rhi2::Capped<Buffer>>,
    pub tl_sem: vk::Semaphore,
    pub count: u64,
    pub bin_sem: Option<vk::Semaphore>,
    pub sync_pool: Arc<SyncPool>,
    waited: bool,
}

impl TaskFuture {
    pub fn new(sync_pool: &Arc<SyncPool>, bin_sem: bool) -> Result<Self, String> {
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
        let bin_sem = if bin_sem {
            let mut pool_mut = match sync_pool.bin_sem_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            match pool_mut.pop() {
                Some(s) => Some(s),
                None => {
                    let sem = unsafe {
                        sync_pool
                            .device_dropper
                            .device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .map_err(|e| format!("creating vk bin sem failed: {e}"))?
                    };
                    Some(sem)
                }
            }
        } else {
            None
        };
        Ok(Self {
            preserve_buffers: vec![],
            tl_sem,
            count,
            bin_sem,
            sync_pool: sync_pool.clone(),
            waited: false,
        })
    }

    pub fn increase_count(&mut self, count: u64) {
        self.count += count;
    }

    pub fn sem_infos(&self) -> Vec<(vk::Semaphore, u64)> {
        let mut out = vec![(self.tl_sem, self.count)];
        if let Some(bs) = self.bin_sem.as_ref() {
            out.push((*bs, 0));
        }
        out
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
        let mut pool_mut = match self.sync_pool.tl_sem_pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        pool_mut.push((self.tl_sem, self.count));

        if let Some(bin_sem) = self.bin_sem.take() {
            let mut pool_mut = match self.sync_pool.bin_sem_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            pool_mut.push(bin_sem);
        }
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
