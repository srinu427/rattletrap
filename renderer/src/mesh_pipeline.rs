use avk12::{
    ash::vk,
    device::Device,
    pipeline::{
        AttachInfo, BindInfo, DSet, FragmentConfig, GraphicsPipeline, GraphicsPipelineCreateInfo,
        VertexAttribute, VertexConfig,
    },
};

use crate::mesh::Vertex;

pub struct MeshPipeline {
    pub gp: GraphicsPipeline,
}

impl MeshPipeline {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let gp_info = GraphicsPipelineCreateInfo::builder()
            .set_layouts(vec![
                vec![BindInfo {
                    type_: vk::DescriptorType::UNIFORM_BUFFER,
                    count: 1,
                }],
                vec![BindInfo {
                    type_: vk::DescriptorType::STORAGE_BUFFER,
                    count: 1,
                }],
                vec![BindInfo {
                    type_: vk::DescriptorType::SAMPLER,
                    count: 1,
                }],
                vec![BindInfo {
                    type_: vk::DescriptorType::SAMPLED_IMAGE,
                    count: 1,
                }],
            ])
            .pc_size(size_of::<u32>())
            .vert_conf(
                VertexConfig::builder()
                    .shader("../renderer/src/shaders/mesh.vert".to_string())
                    .attribs(vec![VertexAttribute::Vec4; 5])
                    .fn_name("main".to_string())
                    .stride(size_of::<Vertex>())
                    .build(),
            )
            .frag_conf(
                FragmentConfig::builder()
                    .shader("../renderer/src/shaders/mesh.frag".to_string())
                    .fn_name("main".to_string())
                    .attachments(vec![AttachInfo {
                        format: device.canvas().info().surf_format().format,
                        clear: true,
                        store: true,
                    }])
                    .build(),
            )
            .build();
        let gp = device.new_graphics_pipeline(gp_info)?;
        Ok(Self { gp })
    }

    pub fn new_cam_dset(&self) -> anyhow::Result<DSet> {
        self.gp.new_set(0)
    }

    pub fn new_model_transforms_dset(&self) -> anyhow::Result<DSet> {
        self.gp.new_set(1)
    }

    pub fn new_sampler_dset(&self) -> anyhow::Result<DSet> {
        self.gp.new_set(2)
    }

    pub fn new_texture_dset(&self) -> anyhow::Result<DSet> {
        self.gp.new_set(3)
    }
}
