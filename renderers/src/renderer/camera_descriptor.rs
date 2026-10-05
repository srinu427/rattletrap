use ash::vk;

use crate::{
    helpers::{
        buffer::GpuObj,
        device::{GpuCommandRecorder, GpuCtx},
        image::GpuDsl,
    },
    renderer::GpuCamera,
};

pub struct CameraDescriptor {
    pub cam_data: GpuObj<GpuCamera>,
    pub dset: vk::DescriptorSet,
}

impl CameraDescriptor {
    pub fn new(ctx: &mut GpuCtx, dsl: &mut GpuDsl) -> anyhow::Result<Self> {
        let cam_data = GpuObj::new(ctx, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        let dset = dsl.get_set(ctx)?;
        unsafe {
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(cam_data.buffer)
                        .range(vk::WHOLE_SIZE)])
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_binding(0)
                    .dst_set(dset)],
                &[],
            );
        }
        Ok(Self { cam_data, dset })
    }

    pub fn update_camera(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        data: &GpuCamera,
    ) -> anyhow::Result<()> {
        self.cam_data.write(ctx, data, cr)?;
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx, dsl: &mut GpuDsl) {
        self.cam_data.destroy(ctx);
        dsl.reclaim(self.dset);
    }
}
