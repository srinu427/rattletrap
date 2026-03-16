use std::sync::{Arc, Mutex};

use ash::vk;
use log::warn;

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

pub struct Fence {
    pub handle: vk::Fence,
    pub pool: Arc<SyncPool>,
}

impl Fence {
    pub fn get(pool: &Arc<SyncPool>, count: usize) -> Result<Vec<Fence>, String> {
        let mut fence_pool = match pool.fence_pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        let mut rem_count = count;
        let mut fences = vec![];
        while let Some(fence) = fence_pool.pop()
            && rem_count > 0
        {
            fences.push(Fence {
                handle: fence,
                pool: pool.clone(),
            });
            rem_count -= 1;
        }
        for _ in 0..rem_count {
            let fence = unsafe {
                pool.device_dropper
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| format!("vk fence create failed: {e}"))?
            };
            let fence = Fence {
                handle: fence,
                pool: pool.clone(),
            };
            fences.push(fence);
        }
        Ok(fences)
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        match unsafe { self.pool.device_dropper.device.reset_fences(&[self.handle]) } {
            Ok(_) => {
                let mut pool = match self.pool.fence_pool.lock() {
                    Ok(fp) => fp,
                    Err(e) => e.into_inner(),
                };
                pool.push(self.handle);
            }
            Err(e) => warn!("fence reset failed: {e}"),
        };
    }
}

pub struct SyncPool {
    pub fence_pool: Arc<Mutex<Vec<vk::Fence>>>,
    pub tl_sem_pool: Arc<Mutex<Vec<(vk::Semaphore, u64)>>>,
    pub bin_sem_pool: Arc<Mutex<Vec<vk::Semaphore>>>,
    pub device_dropper: Arc<DeviceDropper>,
}

impl SyncPool {
    pub fn new(device_dropper: &Arc<DeviceDropper>) -> Self {
        Self {
            fence_pool: Arc::new(Mutex::new(vec![])),
            tl_sem_pool: Arc::new(Mutex::new(vec![])),
            bin_sem_pool: Arc::new(Mutex::new(vec![])),
            device_dropper: device_dropper.clone(),
        }
    }
}

impl Drop for SyncPool {
    fn drop(&mut self) {
        unsafe {
            let mut fence_pool = match self.fence_pool.lock() {
                Ok(p) => p,
                Err(e) => e.into_inner(),
            };
            for fence in fence_pool.drain(..) {
                self.device_dropper.device.destroy_fence(fence, None);
            }
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

pub enum CpuWaitable {
    Fence(Fence),
    TlSem(TlSem),
}

pub struct CpuFuture {
    pub preserve_buffers: Vec<rhi2::Capped<Buffer>>,
    pub inner: CpuWaitable,
    waited: bool,
}

impl CpuFuture {
    pub fn from_tl_sem(sem: TlSem, preserve_buffers: Vec<rhi2::Capped<Buffer>>) -> Self {
        Self {
            preserve_buffers,
            inner: CpuWaitable::TlSem(sem),
            waited: false,
        }
    }

    pub fn from_fence(fence: Fence, preserve_buffers: Vec<rhi2::Capped<Buffer>>) -> Self {
        Self {
            preserve_buffers,
            inner: CpuWaitable::Fence(fence),
            waited: false,
        }
    }

    pub fn increase_count(&mut self, count: u64) {
        match &mut self.inner {
            CpuWaitable::Fence(_) => {}
            CpuWaitable::TlSem(tl_sem) => tl_sem.count += count,
        }
    }
}

impl rhi2::sync::CpuFuture for CpuFuture {
    fn wait(&mut self) {
        if self.waited {
            return;
        }
        match &self.inner {
            CpuWaitable::Fence(fence) => unsafe {
                fence
                    .pool
                    .device_dropper
                    .device
                    .wait_for_fences(&[fence.handle], true, u64::MAX)
                    .inspect_err(|e| warn!("failed waiting for fence: {e}"))
                    .ok();
            },
            CpuWaitable::TlSem(tl_sem) => unsafe {
                tl_sem
                    .pool
                    .device_dropper
                    .device
                    .wait_semaphores(
                        &vk::SemaphoreWaitInfo::default()
                            .semaphores(&[tl_sem.handle])
                            .values(&[tl_sem.count]),
                        u64::MAX,
                    )
                    .inspect_err(|e| warn!("failed waiting for TL Semaphore: {e}"))
                    .ok();
            },
        }

        self.waited = true;
    }
}

pub enum GpuWaitable {
    BinSem(BinSem),
    TlSem(TlSem),
}

pub struct GpuFuture {
    pub preserve_buffers: Vec<rhi2::Capped<Buffer>>,
    pub inner: GpuWaitable,
}

impl GpuFuture {
    pub fn from_tl(tl_sem: TlSem, preserve_buffers: Vec<rhi2::Capped<Buffer>>) -> Self {
        Self {
            preserve_buffers,
            inner: GpuWaitable::TlSem(tl_sem),
        }
    }

    pub fn from_bin(bin_sem: BinSem, preserve_buffers: Vec<rhi2::Capped<Buffer>>) -> Self {
        Self {
            preserve_buffers,
            inner: GpuWaitable::BinSem(bin_sem),
        }
    }

    pub fn get_wait_info(&self) -> (vk::Semaphore, u64) {
        match &self.inner {
            GpuWaitable::BinSem(bin_sem) => (bin_sem.handle, 0),
            GpuWaitable::TlSem(tl_sem) => (tl_sem.handle, tl_sem.count),
        }
    }

    pub fn increase_count(&mut self, count: u64) {
        match &mut self.inner {
            GpuWaitable::BinSem(_) => {}
            GpuWaitable::TlSem(tl_sem) => tl_sem.count += count,
        }
    }
}

impl rhi2::sync::GpuFuture for GpuFuture {}

pub fn rhi2_pipe_stage_to_vk(ps: &rhi2::sync::PipelineStage) -> vk::PipelineStageFlags {
    match ps {
        rhi2::sync::PipelineStage::Top => vk::PipelineStageFlags::TOP_OF_PIPE,
        rhi2::sync::PipelineStage::Vertex => vk::PipelineStageFlags::VERTEX_SHADER,
        rhi2::sync::PipelineStage::DepthTest => vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        rhi2::sync::PipelineStage::Fragment => vk::PipelineStageFlags::FRAGMENT_SHADER,
        rhi2::sync::PipelineStage::AttachWrite => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        rhi2::sync::PipelineStage::Transfer => vk::PipelineStageFlags::TRANSFER,
        rhi2::sync::PipelineStage::End => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    }
}
