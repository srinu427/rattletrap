use std::sync::{Arc, Mutex, PoisonError};

use ash::vk;

use crate::vkraii::device::DeviceDropper;

pub struct SyncPool {
    semaphores: Mutex<Vec<(vk::Semaphore, u64)>>,
    device_d: Arc<DeviceDropper>,
}

impl Drop for SyncPool {
    fn drop(&mut self) {
        for (sem, _) in self
            .semaphores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            unsafe {
                self.device_d.device.destroy_semaphore(*sem, None);
            }
        }
    }
}

pub struct SyncPoolRaii {
    sync_pool_d: Arc<SyncPool>,
}

impl SyncPoolRaii {
    pub fn new(device_d: &Arc<DeviceDropper>) -> Self {
        Self {
            sync_pool_d: Arc::new(SyncPool {
                semaphores: Default::default(),
                device_d: device_d.clone(),
            }),
        }
    }

    pub fn get_sem(&self) -> anyhow::Result<TimelineSemaphore> {
        let sem = self
            .sync_pool_d
            .semaphores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let sem = match sem {
            Some(t) => t,
            None => {
                let sem = unsafe {
                    self.sync_pool_d.device_d.device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(
                            &mut vk::SemaphoreTypeCreateInfo::default()
                                .semaphore_type(vk::SemaphoreType::TIMELINE),
                        ),
                        None,
                    )?
                };
                (sem, 0)
            }
        };
        Ok(TimelineSemaphore {
            semaphore: sem.0,
            value: sem.1,
            sync_pool_d: self.sync_pool_d.clone(),
        })
    }
}

pub struct TimelineSemaphore {
    pub semaphore: vk::Semaphore,
    pub value: u64,
    sync_pool_d: Arc<SyncPool>,
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        self.sync_pool_d
            .semaphores
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((self.semaphore, self.value));
    }
}
