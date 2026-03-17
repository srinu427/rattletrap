use std::sync::{Arc, Mutex};

use ash::vk;

use crate::{buffer::Buffer, device::DeviceDropper};

pub struct TlSem {
    pub handle: vk::Semaphore,
    pub count: u64,
    pub pool: Arc<SyncPool>,
}

impl TlSem {
    pub fn get(pool: &Arc<SyncPool>, count: usize) -> Result<Vec<Self>, String> {
        let mut tl_sem_pool = match pool.tl_sem_pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        let mut rem_count = count;
        let mut tl_sems = vec![];
        while let Some((tl_sem, count)) = tl_sem_pool.pop()
            && rem_count > 0
        {
            tl_sems.push(Self {
                handle: tl_sem,
                count,
                pool: pool.clone(),
            });
            rem_count -= 1;
        }
        let mut tl_sem_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let sem_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut tl_sem_type);
        for _ in 0..rem_count {
            let bin_sem = unsafe {
                pool.device_dropper
                    .device
                    .create_semaphore(&sem_create_info, None)
                    .map_err(|e| format!("vk sem create failed: {e}"))?
            };
            let tl_sem = Self {
                handle: bin_sem,
                count: 0,
                pool: pool.clone(),
            };
            tl_sems.push(tl_sem);
        }
        Ok(tl_sems)
    }
}

impl Drop for TlSem {
    fn drop(&mut self) {
        let mut pool = match self.pool.tl_sem_pool.lock() {
            Ok(fp) => fp,
            Err(e) => e.into_inner(),
        };
        pool.push((self.handle, self.count));
    }
}

pub struct BinSem {
    pub handle: vk::Semaphore,
    pub pool: Arc<SyncPool>,
}

impl BinSem {
    pub fn get(pool: &Arc<SyncPool>, count: usize) -> Result<Vec<Self>, String> {
        let mut bin_sem_pool = match pool.bin_sem_pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        let mut rem_count = count;
        let mut bin_sems = vec![];
        while let Some(bin_sem) = bin_sem_pool.pop()
            && rem_count > 0
        {
            bin_sems.push(Self {
                handle: bin_sem,
                pool: pool.clone(),
            });
            rem_count -= 1;
        }
        for _ in 0..rem_count {
            let bin_sem = unsafe {
                pool.device_dropper
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| format!("vk sem create failed: {e}"))?
            };
            let fence = Self {
                handle: bin_sem,
                pool: pool.clone(),
            };
            bin_sems.push(fence);
        }
        Ok(bin_sems)
    }
}

impl Drop for BinSem {
    fn drop(&mut self) {
        let mut pool = match self.pool.bin_sem_pool.lock() {
            Ok(fp) => fp,
            Err(e) => e.into_inner(),
        };
        pool.push(self.handle);
    }
}

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
    pub inner: TlSem,
    pub bin_sem: Option<BinSem>,
    waited: bool,
}

impl TaskFuture {
    pub fn increase_count(&mut self, count: u64) {
        self.inner.count += count;
    }

    pub fn sem_infos(&self) -> Vec<(vk::Semaphore, u64)> {
        let mut out = vec![(self.inner.handle, self.inner.count)];
        if let Some(bs) = self.bin_sem.as_ref() {
            out.push((bs.handle, 0));
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
            self.inner
                .pool
                .device_dropper
                .device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.inner.handle])
                        .values(&[self.inner.count]),
                    u64::MAX,
                )
                .map_err(|e| format!("failed waiting for TL Semaphore: {e}"))
                .map_err(rhi2::sync::SyncErr::WaitErr)?;
        }

        self.waited = true;
        Ok(())
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
