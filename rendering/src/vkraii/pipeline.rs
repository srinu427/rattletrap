use core::slice;
use std::{
    fs,
    sync::{Arc, Mutex, PoisonError},
};

use ash::vk;
use naga::{
    ShaderStage,
    back::spv,
    front::glsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

use crate::vkraii::device::DeviceDropper;

pub struct ShaderRaii {
    pub module: vk::ShaderModule,
    pub device_d: Arc<DeviceDropper>,
}

impl ShaderRaii {
    pub fn load_glsl_str(
        device_d: &Arc<DeviceDropper>,
        glsl_str: &str,
        stage: ShaderStage,
    ) -> anyhow::Result<Self> {
        let mut frontend = glsl::Frontend::default();
        let options = glsl::Options::from(stage);
        let ir = frontend.parse(&options, &glsl_str)?;
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
        let module_info = validator.validate(&ir)?;
        let spv_words = spv::write_vec(&ir, &module_info, &spv::Options::default(), None)?;
        let module = unsafe {
            device_d.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&spv_words),
                None,
            )?
        };
        Ok(Self {
            module,
            device_d: device_d.clone(),
        })
    }

    pub fn load_glsl(device_d: &Arc<DeviceDropper>, path: &str) -> anyhow::Result<Self> {
        let shader_stage = if path.ends_with(".vert") {
            ShaderStage::Vertex
        } else if path.ends_with(".frag") {
            ShaderStage::Fragment
        } else {
            anyhow::bail!("unknown shader ext type");
        };
        let inp_str = fs::read_to_string(path)?;
        Self::load_glsl_str(device_d, &inp_str, shader_stage)
    }
}

impl Drop for ShaderRaii {
    fn drop(&mut self) {
        unsafe {
            self.device_d
                .device
                .destroy_shader_module(self.module, None);
        }
    }
}

struct DescriptorPoolDropper {
    pool: vk::DescriptorPool,
    device_d: Arc<DeviceDropper>,
}

impl Drop for DescriptorPoolDropper {
    fn drop(&mut self) {
        unsafe {
            self.device_d
                .device
                .destroy_descriptor_pool(self.pool, None);
        }
    }
}

pub struct DescriptorSetLayoutRaii {
    pub bindings: Vec<(vk::DescriptorType, u32)>,
    pub layout: vk::DescriptorSetLayout,
    pub alloc_batch_size: u32,
    pool: Arc<Mutex<Vec<(vk::DescriptorSet, Arc<DescriptorPoolDropper>)>>>,
    device_d: Arc<DeviceDropper>,
}

impl DescriptorSetLayoutRaii {
    pub fn new(
        device_d: &Arc<DeviceDropper>,
        create_info: &vk::DescriptorSetLayoutCreateInfo,
        alloc_batch_size: u32,
    ) -> anyhow::Result<Self> {
        let layout = unsafe {
            device_d
                .device
                .create_descriptor_set_layout(create_info, None)?
        };
        let bindings: Vec<_> = unsafe {
            slice::from_raw_parts(create_info.p_bindings, create_info.binding_count as _)
        }
        .iter()
        .map(|b| (b.descriptor_type, b.descriptor_count))
        .collect();
        Ok(Self {
            bindings,
            layout,
            alloc_batch_size,
            pool: Default::default(),
            device_d: device_d.clone(),
        })
    }

    pub fn get_set(&self) -> anyhow::Result<DescriptorSetRaii> {
        let mut pool_mut = self.pool.lock().unwrap_or_else(PoisonError::into_inner);
        let set = pool_mut.pop();
        let set = match set {
            Some(t) => t,
            None => {
                let pool = unsafe {
                    self.device_d
                        .device
                        .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default(), None)?
                };
                let sets = unsafe {
                    self.device_d.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(pool)
                            .set_layouts(&vec![self.layout; self.alloc_batch_size as _]),
                    )?
                };
                let new_pool_d = Arc::new(DescriptorPoolDropper {
                    pool,
                    device_d: self.device_d.clone(),
                });
                let new_pool_content: Vec<_> =
                    sets[1..].iter().map(|s| (*s, new_pool_d.clone())).collect();
                pool_mut.extend(new_pool_content);
                (sets[0], new_pool_d.clone())
            }
        };
        Ok(DescriptorSetRaii {
            set: set.0,
            pool_d: set.1,
            pool: self.pool.clone(),
        })
    }
}

pub struct DescriptorSetRaii {
    pub set: vk::DescriptorSet,
    pool_d: Arc<DescriptorPoolDropper>,
    pool: Arc<Mutex<Vec<(vk::DescriptorSet, Arc<DescriptorPoolDropper>)>>>,
}

impl Drop for DescriptorSetRaii {
    fn drop(&mut self) {
        self.pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((self.set, self.pool_d.clone()));
    }
}
