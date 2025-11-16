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
