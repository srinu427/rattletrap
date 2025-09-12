use std::sync::Arc;

use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(Debug, thiserror::Error)]
pub enum DescriptorPoolError {
    #[error("Descriptor pool creation error: {0}")]
    CreateError(vk::Result),
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct DescriptorPool {
    #[get_copy = "pub"]
    pool: vk::DescriptorPool,
    #[get = "pub"]
    device: Arc<LogicalDevice>,
}

impl DescriptorPool {
    pub fn new(
        device: Arc<LogicalDevice>,
        pool_sizes: &[(vk::DescriptorType, u32)],
        max_sets: u32,
    ) -> Result<Self, DescriptorPoolError> {
        let vk_pool_sizes: Vec<vk::DescriptorPoolSize> = pool_sizes
            .iter()
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *count,
            })
            .collect();

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&vk_pool_sizes)
            .max_sets(max_sets);

        let pool = unsafe {
            device
                .device()
                .create_descriptor_pool(&create_info, None)
                .map_err(DescriptorPoolError::CreateError)?
        };

        Ok(Self { pool, device })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_descriptor_pool(self.pool, None);
        }
    }
}
