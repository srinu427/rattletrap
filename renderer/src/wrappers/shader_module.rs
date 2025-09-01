use ash::vk;

use crate::wrappers::logical_device::LogicalDevice;

#[derive(Debug, thiserror::Error)]
pub enum ShaderModuleError {
    #[error("Shader module creation error: {0}")]
    CreateError(vk::Result),
}

pub fn make_shader_module(
    device: &LogicalDevice,
    code: &[u8],
) -> Result<vk::ShaderModule, ShaderModuleError> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(code));

    unsafe {
        device
            .device()
            .create_shader_module(&create_info, None)
            .map_err(ShaderModuleError::CreateError)
    }
}
