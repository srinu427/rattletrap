use std::sync::Arc;

use hashbrown::HashMap;

use crate::renderer::material::Material;
use crate::renderer::mesh::Mesh;

pub struct DrawableInfo {}

pub struct AssetManager {
    meshes: Vec<Mesh>,
    mesh_names: HashMap<Arc<String>, usize>,
    v_buffers: Vec<rhi::Buffer>,
    v_stage_buffers: Vec<rhi::Buffer>,
    i_buffers: Vec<rhi::Buffer>,
    i_stage_buffers: Vec<rhi::Buffer>,
    materials: Vec<Material>,
    material_names: HashMap<Arc<String>, usize>,
    material_dset: rhi::DSet,
}
