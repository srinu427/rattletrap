use std::sync::Arc;

use ash::vk;

use crate::{
    buffer::Buffer,
    graphics_pipeline::DPool,
    image::{Image, ImageView},
};

pub struct ShaderSet {
    pub handle: vk::DescriptorSet,
    pub pool: Arc<DPool>,
}

impl rhi2::shader::ShaderSet for ShaderSet {
    type BType = Buffer;

    type IType = Image;

    type IVType = ImageView;

    fn update_binding(
        &mut self,
        binding: usize,
        data: rhi2::shader::ShaderSetData<Self::BType, Self::IVType>,
    ) {
        match data {
            rhi2::shader::ShaderSetData::UniformBuffer(cappeds) => unsafe {
                let buffer_infos: Vec<_> = cappeds
                    .iter()
                    .map(|cb| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(cb.as_ref().handle)
                            .range(vk::WHOLE_SIZE)
                    })
                    .collect();
                self.pool.device_dropper.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.handle)
                        .dst_binding(binding as _)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(buffer_infos.len() as _)
                        .buffer_info(&buffer_infos)],
                    &[],
                );
            },
            rhi2::shader::ShaderSetData::StorageBuffer(cappeds) => unsafe {
                let buffer_infos: Vec<_> = cappeds
                    .iter()
                    .map(|cb| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(cb.as_ref().handle)
                            .range(vk::WHOLE_SIZE)
                    })
                    .collect();
                self.pool.device_dropper.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.handle)
                        .dst_binding(binding as _)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(buffer_infos.len() as _)
                        .buffer_info(&buffer_infos)],
                    &[],
                );
            },
            rhi2::shader::ShaderSetData::Sampler2D(cappeds) => unsafe {
                let image_infos: Vec<_> = cappeds
                    .iter()
                    .map(|ci| {
                        vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(ci.as_ref().handle)
                    })
                    .collect();
                self.pool.device_dropper.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.handle)
                        .dst_binding(binding as _)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(image_infos.len() as _)
                        .image_info(&image_infos)],
                    &[],
                );
            },
        };
        todo!()
    }
}
