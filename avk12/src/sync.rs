use std::sync::{Arc, Mutex, PoisonError};

use anyhow::Context;
use ash::vk;

use crate::device::DeviceDropper;

#[derive(Debug, Clone)]
pub(crate) struct SemVal {
    pub(crate) handle: vk::Semaphore,
    pub(crate) val: u64,
}

pub(crate) struct SemPoolDropper {
    pub(crate) sem_vals: Mutex<Vec<SemVal>>,
    pub(crate) dd: Arc<DeviceDropper>,
}

impl Drop for SemPoolDropper {
    fn drop(&mut self) {
        unsafe {
            for sem_val in self
                .sem_vals
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .drain(..)
            {
                self.dd.device.destroy_semaphore(sem_val.handle, None);
            }
        }
    }
}

pub(crate) struct SemPool {
    dropper: Arc<SemPoolDropper>,
}

impl SemPool {
    pub(crate) fn new(dd: &Arc<DeviceDropper>) -> Self {
        Self {
            dropper: Arc::new(SemPoolDropper {
                sem_vals: Mutex::new(vec![]),
                dd: dd.clone(),
            }),
        }
    }

    pub(crate) fn get_sem(&self) -> anyhow::Result<Sem> {
        let ex_sem_val = self
            .dropper
            .sem_vals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let sem_val = match ex_sem_val {
            Some(sem_val) => sem_val,
            None => {
                let sem_handle = unsafe {
                    self.dropper
                        .dd
                        .device
                        .create_semaphore(
                            &vk::SemaphoreCreateInfo::default().push_next(
                                &mut vk::SemaphoreTypeCreateInfo::default()
                                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                                    .initial_value(0),
                            ),
                            None,
                        )
                        .context("semaphore creation failed")?
                };
                SemVal {
                    handle: sem_handle,
                    val: 0,
                }
            }
        };
        Ok(Sem {
            sem_val,
            dropper: self.dropper.clone(),
        })
    }
}

pub(crate) enum WaitResult {
    Success,
    Timeout,
    Error(String),
}

pub(crate) struct Sem {
    pub(crate) sem_val: SemVal,
    pub(crate) dropper: Arc<SemPoolDropper>,
}

impl Sem {
    pub fn wait(&self, timeout: u64) -> WaitResult {
        let wait_res = unsafe {
            self.dropper.dd.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[self.sem_val.handle])
                    .values(&[self.sem_val.val]),
                timeout,
            )
        };
        match wait_res {
            Ok(_) => WaitResult::Success,
            Err(e) => match e {
                vk::Result::TIMEOUT => WaitResult::Timeout,
                _ => WaitResult::Error(format!("wait for semaphore failed: {e}")),
            },
        }
    }
}

impl Drop for Sem {
    fn drop(&mut self) {
        self.dropper
            .sem_vals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.sem_val.clone());
    }
}

pub(crate) struct FencePoolDropper {
    pool: Mutex<Vec<vk::Fence>>,
    dd: Arc<DeviceDropper>,
}

impl Drop for FencePoolDropper {
    fn drop(&mut self) {
        for fence in self
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain(..)
        {
            unsafe {
                self.dd.device.destroy_fence(fence, None);
            }
        }
    }
}

pub(crate) struct FencePool {
    dropper: Arc<FencePoolDropper>,
}

impl FencePool {
    pub(crate) fn new(dd: &Arc<DeviceDropper>) -> Self {
        Self {
            dropper: Arc::new(FencePoolDropper {
                pool: Mutex::new(vec![]),
                dd: dd.clone(),
            }),
        }
    }

    pub(crate) fn get_fence(&self) -> anyhow::Result<Fence> {
        let ex_fence = self
            .dropper
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let handle = match ex_fence {
            Some(handle) => handle,
            None => {
                let handle = unsafe {
                    self.dropper
                        .dd
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .context("fence creation failed")?
                };
                handle
            }
        };
        Ok(Fence {
            handle,
            fpd: self.dropper.clone(),
        })
    }
}

pub(crate) struct Fence {
    pub(crate) handle: vk::Fence,
    fpd: Arc<FencePoolDropper>,
}

impl Fence {
    fn reset(&self) -> anyhow::Result<()> {
        unsafe {
            self.fpd
                .dd
                .device
                .reset_fences(&[self.handle])
                .context("reset fence failed")
        }
    }

    pub(crate) fn wait(&self, timeout: u64) -> WaitResult {
        let wait_res = unsafe {
            self.fpd
                .dd
                .device
                .wait_for_fences(&[self.handle], true, timeout)
        };
        match wait_res {
            Ok(_) => {
                let _ = self.reset();
                WaitResult::Success
            }
            Err(e) => match e {
                vk::Result::TIMEOUT => WaitResult::Timeout,
                _ => WaitResult::Error(format!("wait for fence failed: {e}")),
            },
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        let _ = self.reset();
        self.fpd
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.handle);
    }
}
