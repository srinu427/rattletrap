use ash::vk;

use crate::make_init_struct_copy;

make_init_struct_copy!(
    InitFence,
    vk::Fence,
    self,
    self.device.destroy_fence(self.inner, None)
);

pub fn create_fence(device: &'_ ash::Device) -> Result<InitFence<'_>, vk::Result> {
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    Ok(InitFence {
        drop: true,
        inner: fence,
        device,
    })
}

pub fn wait_for_fences(
    device: &ash::Device,
    fences: &[vk::Fence],
    timeout: Option<u64>,
) -> Result<(), vk::Result> {
    unsafe { device.wait_for_fences(fences, true, timeout.unwrap_or(u64::MAX)) }
}

pub fn reset_fences(device: &ash::Device, fences: &[vk::Fence]) -> Result<(), vk::Result> {
    unsafe { device.reset_fences(&fences) }
}
