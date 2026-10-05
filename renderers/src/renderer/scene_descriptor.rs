use ash::vk;

use crate::{
    Material,
    helpers::{
        buffer::GpuVec,
        device::{GpuCommandRecorder, GpuCtx},
        image::GpuDsl,
    },
    renderer::{GpuLightData, GpuMeshInfo},
};

pub struct SceneDescriptor {
    pub mesh_infos: GpuVec<GpuMeshInfo>,
    pub materials: GpuVec<Material>,
    pub lights: GpuVec<GpuLightData>,
    pub dset: vk::DescriptorSet,
}

impl SceneDescriptor {
    pub fn new(ctx: &mut GpuCtx, dsl: &mut GpuDsl) -> anyhow::Result<Self> {
        let mesh_infos = GpuVec::new(ctx, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let materials = GpuVec::new(ctx, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let lights = GpuVec::new(ctx, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let dset = dsl.get_set(ctx)?;
        unsafe {
            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(mesh_infos.buffer)
                            .range((mesh_infos.len.max(1) * size_of::<GpuMeshInfo>()) as _)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(0)
                        .dst_set(dset),
                    vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(materials.buffer)
                            .range((materials.len.max(1) * size_of::<Material>()) as _)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(1)
                        .dst_set(dset),
                    vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(lights.buffer)
                            .range((lights.len.max(1) * size_of::<GpuLightData>()) as _)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(2)
                        .dst_set(dset),
                ],
                &[],
            );
        }
        Ok(Self {
            mesh_infos,
            materials,
            lights,
            dset,
        })
    }

    pub fn udpate_mesh_infos(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        data: &[GpuMeshInfo],
    ) -> anyhow::Result<()> {
        let old_buf = self.mesh_infos.buffer;
        let old_len = self.mesh_infos.len;
        self.mesh_infos.write(ctx, 0, data, cr)?;
        if self.mesh_infos.buffer != old_buf || self.mesh_infos.len != old_len {
            unsafe {
                ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(self.mesh_infos.buffer)
                            .range((self.mesh_infos.len.max(1) * size_of::<GpuMeshInfo>()) as _)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(0)
                        .dst_set(self.dset)],
                    &[],
                );
            }
        }
        Ok(())
    }

    pub fn udpate_materials(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        data: &[Material],
    ) -> anyhow::Result<()> {
        let old_buf = self.materials.buffer;
        let old_len = self.materials.len;
        self.materials.write(ctx, 0, data, cr)?;
        if self.materials.buffer != old_buf || self.materials.len != old_len {
            unsafe {
                ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(self.materials.buffer)
                            .range((self.materials.len.max(1) * size_of::<Material>()) as _)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(1)
                        .dst_set(self.dset)],
                    &[],
                );
            }
        }
        Ok(())
    }

    pub fn update_lights(
        &mut self,
        ctx: &mut GpuCtx,
        cr: &mut GpuCommandRecorder,
        data: &[GpuLightData],
    ) -> anyhow::Result<()> {
        let old_buf = self.lights.buffer;
        let old_len = self.lights.len;
        self.lights.write(ctx, 0, data, cr)?;
        if self.lights.buffer != old_buf || self.lights.len != old_len {
            unsafe {
                ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(self.lights.buffer)
                            .range((self.lights.len.max(1) * size_of::<GpuLightData>()) as _)])
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(2)
                        .dst_set(self.dset)],
                    &[],
                );
            }
        }
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx, dsl: &mut GpuDsl) {
        self.mesh_infos.destroy(ctx);
        self.materials.destroy(ctx);
        self.lights.destroy(ctx);
        dsl.reclaim(self.dset);
    }
}
