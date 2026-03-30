use std::sync::Arc;

use ash::vk;
use rhi2::shader::{ShaderSetData, ShaderSetInfo};

use crate::{
    buffer::Buffer,
    device::DeviceDropper,
    image::{Image, ImageView, Sampler},
};

pub struct DPoolDropper {
    pub handle: vk::DescriptorPool,
    pub device_dropper: Arc<DeviceDropper>,
}

impl Drop for DPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.device_dropper
                .device
                .destroy_descriptor_pool(self.handle, None);
        }
    }
}

pub trait SsiVk {
    fn vk_dp_size(&self) -> vk::DescriptorPoolSize;
    fn vk_desc_type(&self) -> vk::DescriptorType;
    fn vk_desc_count(&self) -> usize;
    fn vk_binding_info(&self, bind_idx: usize) -> vk::DescriptorSetLayoutBinding<'_>;
}

impl SsiVk for ShaderSetInfo {
    fn vk_dp_size(&self) -> vk::DescriptorPoolSize {
        match self {
            ShaderSetInfo::UniformBuffer(count) => vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(*count as _),
            ShaderSetInfo::StorageBuffer(count) => vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(*count as _),
            ShaderSetInfo::Sampler2D(count) => vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(*count as _),
        }
    }

    fn vk_desc_type(&self) -> vk::DescriptorType {
        match self {
            ShaderSetInfo::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            ShaderSetInfo::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            ShaderSetInfo::Sampler2D(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }

    fn vk_desc_count(&self) -> usize {
        match self {
            rhi2::shader::ShaderSetInfo::UniformBuffer(c) => *c,
            rhi2::shader::ShaderSetInfo::StorageBuffer(c) => *c,
            rhi2::shader::ShaderSetInfo::Sampler2D(c) => *c,
        }
    }

    fn vk_binding_info(&self, bind_idx: usize) -> vk::DescriptorSetLayoutBinding<'_> {
        match self {
            ShaderSetInfo::UniformBuffer(c) => vk::DescriptorSetLayoutBinding::default()
                .binding(bind_idx as _)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(*c as _),
            ShaderSetInfo::StorageBuffer(c) => vk::DescriptorSetLayoutBinding::default()
                .binding(bind_idx as _)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(*c as _),
            ShaderSetInfo::Sampler2D(c) => vk::DescriptorSetLayoutBinding::default()
                .binding(bind_idx as _)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(*c as _),
        }
    }
}

pub struct DPool {
    pub bindings: Vec<rhi2::shader::ShaderSetInfo>,
    pub layout: vk::DescriptorSetLayout,
    pub pool: Arc<DPoolDropper>,
}

impl DPool {
    pub fn new(
        device_dropper: &Arc<DeviceDropper>,
        bindings: Vec<rhi2::shader::ShaderSetInfo>,
    ) -> Result<Self, String> {
        let dsl_bindings: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| b.vk_binding_info(i))
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings);
        let layout = unsafe {
            device_dropper
                .device
                .create_descriptor_set_layout(&dsl_info, None)
                .map_err(|e| format!("error creating dsl: {e}"))?
        };
        let pool_sizes: Vec<_> = bindings.iter().map(|b| b.vk_dp_size()).collect();
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&pool_sizes)
            .max_sets(128);
        let pool = unsafe {
            device_dropper
                .device
                .create_descriptor_pool(&pool_create_info, None)
                .map_err(|e| format!("create Desc Pool failed: {e}"))?
        };
        let pool = DPoolDropper {
            handle: pool,
            device_dropper: device_dropper.clone(),
        };
        Ok(DPool {
            bindings,
            layout,
            pool: Arc::new(pool),
        })
    }

    pub fn new_ss(
        &mut self,
        data: Vec<ShaderSetData<Buffer, ImageView, Sampler>>,
    ) -> Result<ShaderSet, String> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool.handle)
            .set_layouts(core::slice::from_ref(&self.layout));
        let dset = unsafe {
            match self
                .pool
                .device_dropper
                .device
                .allocate_descriptor_sets(&alloc_info)
            {
                Ok(mut dset) => dset.remove(0),
                Err(e) => match e {
                    vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                        let pool_sizes: Vec<_> =
                            self.bindings.iter().map(|b| b.vk_dp_size()).collect();
                        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
                            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                            .pool_sizes(&pool_sizes)
                            .max_sets(128);
                        let pool = self
                            .pool
                            .device_dropper
                            .device
                            .create_descriptor_pool(&pool_create_info, None)
                            .map_err(|e| format!("create Desc Pool failed: {e}"))?;
                        let pool = DPoolDropper {
                            handle: pool,
                            device_dropper: self.pool.device_dropper.clone(),
                        };
                        self.pool = Arc::new(pool);
                        let dset = self
                            .pool
                            .device_dropper
                            .device
                            .allocate_descriptor_sets(&alloc_info)
                            .map_err(|e| format!("create Desc Set failed: {e}"))?
                            .remove(0);
                        dset
                    }
                    _ => return Err(format!("create Desc Set failed: {e}")),
                },
            }
        };
        let mut update_infos = vec![];
        for ssd in &data {
            match ssd {
                ShaderSetData::UniformBuffer(cappeds) => {
                    let buf_infos: Vec<_> = cappeds
                        .iter()
                        .map(|b| {
                            vk::DescriptorBufferInfo::default()
                                .buffer(b.as_ref().handle)
                                .range(vk::WHOLE_SIZE)
                        })
                        .collect();
                    update_infos.push((Some(buf_infos), None, vk::DescriptorType::UNIFORM_BUFFER));
                }
                ShaderSetData::StorageBuffer(cappeds) => {
                    let buf_infos: Vec<_> = cappeds
                        .iter()
                        .map(|b| {
                            vk::DescriptorBufferInfo::default()
                                .buffer(b.as_ref().handle)
                                .range(vk::WHOLE_SIZE)
                        })
                        .collect();
                    update_infos.push((Some(buf_infos), None, vk::DescriptorType::STORAGE_BUFFER));
                }
                ShaderSetData::Sampler2D(cappeds) => {
                    let img_infos: Vec<_> = cappeds
                        .iter()
                        .map(|(iv, s)| {
                            vk::DescriptorImageInfo::default()
                                .image_view(iv.as_ref().handle)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(s.as_ref().handle)
                        })
                        .collect();
                    update_infos.push((
                        None,
                        Some(img_infos),
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ));
                }
            }
        }
        let mut write_infos = vec![];
        for (bid, upd_info) in update_infos.iter().enumerate() {
            let mut write_info = vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .descriptor_type(upd_info.2)
                .dst_binding(bid as _);
            if let Some(b_infos) = &upd_info.0 {
                write_info = write_info
                    .descriptor_count(b_infos.len() as _)
                    .buffer_info(b_infos);
            }
            if let Some(i_infos) = &upd_info.1 {
                write_info = write_info
                    .descriptor_count(i_infos.len() as _)
                    .image_info(i_infos);
            }
            write_infos.push(write_info);
        }

        unsafe {
            self.pool
                .device_dropper
                .device
                .update_descriptor_sets(&write_infos, &[]);
        }
        let dset = ShaderSet {
            data,
            handle: dset,
            pool: self.pool.clone(),
        };
        Ok(dset)
    }
}

impl Drop for DPool {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device_dropper
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

pub struct ShaderSet {
    pub data: Vec<ShaderSetData<Buffer, ImageView, Sampler>>,
    pub handle: vk::DescriptorSet,
    pub pool: Arc<DPoolDropper>,
}

impl rhi2::shader::ShaderSet for ShaderSet {
    type B = Buffer;

    type I = Image;

    type IV = ImageView;

    type S = Sampler;

    fn update_binding_data(
        &mut self,
        binding: usize,
        data: ShaderSetData<Self::B, Self::IV, Self::S>,
    ) {
        let mut b_infos = None;
        let mut i_infos = None;
        let d_type = match &data {
            ShaderSetData::UniformBuffer(cappeds) => {
                b_infos = Some(
                    cappeds
                        .iter()
                        .map(|b| {
                            vk::DescriptorBufferInfo::default()
                                .buffer(b.as_ref().handle)
                                .range(vk::WHOLE_SIZE)
                        })
                        .collect::<Vec<_>>(),
                );
                vk::DescriptorType::UNIFORM_BUFFER
            }
            ShaderSetData::StorageBuffer(cappeds) => {
                b_infos = Some(
                    cappeds
                        .iter()
                        .map(|b| {
                            vk::DescriptorBufferInfo::default()
                                .buffer(b.as_ref().handle)
                                .range(vk::WHOLE_SIZE)
                        })
                        .collect::<Vec<_>>(),
                );
                vk::DescriptorType::STORAGE_BUFFER
            }
            ShaderSetData::Sampler2D(cappeds) => {
                i_infos = Some(
                    cappeds
                        .iter()
                        .map(|(iv, s)| {
                            vk::DescriptorImageInfo::default()
                                .image_view(iv.as_ref().handle)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(s.as_ref().handle)
                        })
                        .collect::<Vec<_>>(),
                );
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
        };
        let mut write_info = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .descriptor_type(d_type)
            .dst_binding(binding as _);
        if let Some(b_infos) = &b_infos {
            write_info = write_info
                .descriptor_count(b_infos.len() as _)
                .buffer_info(b_infos);
        }
        if let Some(i_infos) = &i_infos {
            write_info = write_info
                .descriptor_count(i_infos.len() as _)
                .image_info(i_infos);
        }
        unsafe {
            self.pool
                .device_dropper
                .device
                .update_descriptor_sets(&[write_info], &[]);
        }
        self.data[binding] = data;
    }
}
