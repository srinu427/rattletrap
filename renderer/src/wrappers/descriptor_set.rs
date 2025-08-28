use std::sync::Arc;

use ash::vk;

use crate::wrappers::descriptor_pool::DescriptorPool;

pub struct DescriptorSet {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device()
                .device()
                .free_descriptor_sets(self.pool.pool(), &[self.set])
                .inspect_err(|e| eprintln!("Failed to free descriptor set: {e}"))
                .ok();
        }
    }
}
