use gpu_allocator::vulkan::{Allocation, Allocator};

pub mod buffer;
pub mod device;
pub mod image;
pub mod swapchain;

fn free_allocation_logged(allocator: &mut Allocator, allocation: Allocation) {
    if let Err(e) = allocator.free(allocation) {
        log::error!("Freeing GPU allocation failed: {e}")
    }
}
