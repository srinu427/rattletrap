use ash::vk;

use crate::make_init_struct_copy;

make_init_struct_copy!(
    InitCommandPool,
    vk::CommandPool,
    self,
    self.device.destroy_command_pool(self.inner, None)
);

pub fn create_command_pool(
    device: &'_ ash::Device,
    queue_family: u32,
) -> Result<InitCommandPool<'_>, vk::Result> {
    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
            None,
        )?
    };
    Ok(InitCommandPool {
        drop: true,
        inner: command_pool,
        device,
    })
}

pub fn allocate_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(count),
        )
    }
}

pub fn begin_cmd_buffer(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    one_time: bool,
) -> Result<(), vk::Result> {
    let flags = if one_time {
        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
    } else {
        vk::CommandBufferUsageFlags::empty()
    };
    unsafe {
        device.begin_command_buffer(
            cmd_buffer,
            &vk::CommandBufferBeginInfo::default().flags(flags),
        )?;
    }

    Ok(())
}

pub fn end_cmd_buffer(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
) -> Result<(), vk::Result> {
    unsafe {
        device.end_command_buffer(cmd_buffer)?;
    }
    Ok(())
}
