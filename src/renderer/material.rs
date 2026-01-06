use std::sync::Arc;

use hashbrown::HashMap;
use image::EncodableLayout;

pub struct Material {
    name: Arc<String>,
    sampler: Arc<rhi::Sampler>,
    albedo: rhi::ImageView,
}

impl Material {
    fn load_image(device: &rhi::Device, path: &str) -> anyhow::Result<rhi::Image> {
        let image_data = image::open(path)?;
        let image_data_rgba = image_data.to_rgba8();
        let image_data_bytes = image_data_rgba.as_bytes();
        let mut stage_buffer = device.create_buffer(
            image_data_bytes.len() as _,
            rhi::BufferFlags::CopySrc.into(),
            rhi::MemLocation::CpuToGpu,
        )?;
        stage_buffer.write_data(&image_data_bytes)?;
        let image = device.create_image(
            rhi::Dimension::D2,
            rhi::Format::Rgba8Srgb,
            image_data.width(),
            image_data.height(),
            1,
            1,
            rhi::ImageUsage::CopyDst | rhi::ImageUsage::CopySrc | rhi::ImageUsage::Sampled,
            rhi::MemLocation::Gpu,
        )?;
        let cmd_buffer = device.graphics_queue().create_command_buffer()?;
        let mut encoder = cmd_buffer.encoder()?;
        encoder.set_last_image_access(&image, rhi::ImageAccess::Undefined, 0..1, 0..1);
        encoder.copy_buffer_to_image(&stage_buffer, &image, 0..1, 0);
        encoder.set_last_image_access(
            &image,
            rhi::ImageAccess::Shader(rhi::RWAccess::Read),
            0..1,
            0..1,
        );
        encoder.finalize()?;
        let semaphore = device.create_semaphore(false)?;
        cmd_buffer.submit(vec![], vec![semaphore.submit_info(1)])?;
        semaphore.wait_for(1, None)?;
        drop(stage_buffer);
        Ok(image)
    }

    pub fn new(
        device: &rhi::Device,
        path: &str,
        sampler: &Arc<rhi::Sampler>,
    ) -> anyhow::Result<Self> {
        let albedo_path = format!("{path}/albedo.png");
        let albedo_image = Self::load_image(device, &albedo_path)?;
        let albedo = albedo_image.create_view(rhi::ViewDimension::D2, 0..1, 0..1)?;
        Ok(Self {
            name: Arc::new(path.to_string()),
            sampler: sampler.clone(),
            albedo,
        })
    }
}

pub struct MaterialSet {
    pub dset: rhi::DSet,
    binding_id: u32,
    textures: Vec<Material>,
    tex_name_id: HashMap<Arc<String>, usize>,
}

impl MaterialSet {
    pub fn new(dset: rhi::DSet, binding_id: u32) -> anyhow::Result<Self> {
        Ok(Self {
            dset,
            binding_id,
            textures: vec![],
            tex_name_id: HashMap::new(),
        })
    }

    fn update_dset(&mut self) {
        self.dset.write_binding_full(
            self.binding_id,
            rhi::DBindingData::Sampler2d(
                self.textures
                    .iter()
                    .map(|t| (&t.albedo, t.sampler.as_ref()))
                    .collect(),
            ),
        );
    }

    pub fn get_id(&self, s: &String) -> Option<usize> {
        self.tex_name_id.get(s).copied()
    }

    pub fn add(&mut self, mat: Material) {
        if !self.tex_name_id.contains_key(&mat.name) {
            self.tex_name_id
                .insert(mat.name.clone(), self.textures.len());
            self.textures.push(mat);
            self.update_dset();
        }
    }

    pub fn remove(&mut self, name: &String) {
        let Some(tex_id) = self.tex_name_id.remove(name) else {
            return;
        };
        if tex_id == self.textures.len() - 1 {
            self.textures.pop();
        } else {
            self.textures.swap_remove(tex_id);
            if let Some(moved) = self.textures.get(tex_id) {
                self.tex_name_id.insert(moved.name.clone(), tex_id);
            }
        }
        self.update_dset();
    }
}
