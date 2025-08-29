use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

pub fn make_shader_module(
    device: &LogicalDevice,
    code: &[u8],
) -> Result<vk::ShaderModule, vk::Result> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(code));

    unsafe { device.device().create_shader_module(&create_info, None) }
}
