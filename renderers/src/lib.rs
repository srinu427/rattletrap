use serde::{Deserialize, Serialize};

pub type Color3 = [u8; 3];
pub type Color4 = [u8; 4];
pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];
pub type Mat4 = [[f32; 4]; 4];

pub mod vk12;

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u16>,
}

impl Mesh {
    pub fn merge(meshes: Vec<Self>) -> Self {
        let mut vert_offset = 0usize;
        let mut vertices = vec![];
        let mut indices = vec![];
        for mesh in meshes {
            let verts_len = mesh.vertices.len();
            vertices.extend(mesh.vertices);
            for indx in mesh.indices {
                indices.push(vert_offset as u16 + indx);
            }
            vert_offset += verts_len;
        }
        Self { vertices, indices }
    }

    pub fn new_rectangle(c: glam::Vec3, x: glam::Vec3, y: glam::Vec3) -> Self {
        let vertices = [c + x + y, c - x + y, c - x - y, c + x - y]
            .iter()
            .map(|v| MeshVertex { pos: v.to_array() })
            .collect();
        let indices = vec![0, 1, 2, 2, 3, 0];
        Self { vertices, indices }
    }

    pub fn new_cube(c: glam::Vec3, x: glam::Vec3, y: glam::Vec3, h: f32) -> Self {
        let z = h * x.cross(y).normalize();
        let rects = vec![
            Self::new_rectangle(c + x, y, z),
            Self::new_rectangle(c - x, z, y),
            Self::new_rectangle(c + y, z, x),
            Self::new_rectangle(c - y, x, z),
            Self::new_rectangle(c + z, x, y),
            Self::new_rectangle(c - z, y, x),
        ];
        Self::merge(rects)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, bytemuck::NoUninit)]
pub struct Material {
    pub base_weight: f32,
    pub base_color: Color4,
    pub base_metalness: f32,
    pub base_diffuse_roughness: f32,
    pub specular_weight: f32,
    pub specular_color: Color4,
    pub specular_roughness: f32,
    pub specular_roughness_anisotropy: f32,
    pub specular_ior: f32,
    pub transmission_weight: f32,
    pub transmission_color: Color4,
    pub transmission_depth: f32,
    pub transmission_scatter: Vec3,
    pub transmission_scatter_anisotropy: f32,
    pub transmission_dispersion_scale: f32,
    pub transmission_dispersion_abbe_number: f32,
    pub subsurface_weight: f32,
    pub subsurface_color: Color4,
    pub subsurface_radius: f32,
    pub subsurface_radius_scale: Vec3,
    pub subsurface_scatter_anisotropy: f32,
    pub coat_weight: f32,
    pub coat_color: Color4,
    pub coat_roughness: f32,
    pub coat_roughness_anisotropy: f32,
    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub fuzz_weight: f32,
    pub fuzz_color: Color4,
    pub fuzz_roughness: f32,
    pub emission_luminance: f32,
    pub emission_color: Color4,
    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub geometry_opacity: f32,
    pub geometry_thin_walled: u32,
    // pub geometry_normal: [f32; 3],
    // pub geometry_tangent: [f32; 3],
    // pub geometry_coat_normal: [f32; 3],
    // pub geometry_coat_tangent: [f32; 3],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            base_color: [204; 4],
            base_metalness: 0.0,
            base_diffuse_roughness: 0.0,
            specular_weight: 1.0,
            specular_color: [255; 4],
            specular_roughness: 0.3,
            specular_roughness_anisotropy: 0.0,
            specular_ior: 1.5,
            transmission_weight: 0.0,
            transmission_color: [255; 4],
            transmission_depth: 0.0,
            transmission_scatter: [0.0; 3],
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_scale: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            subsurface_weight: 0.0,
            subsurface_color: [204; 4],
            subsurface_radius: 1.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25],
            subsurface_scatter_anisotropy: 0.0,
            coat_weight: 0.0,
            coat_color: [255; 4],
            coat_roughness: 0.0,
            coat_roughness_anisotropy: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            fuzz_weight: 0.0,
            fuzz_color: [255; 4],
            fuzz_roughness: 0.5,
            emission_luminance: 0.0,
            emission_color: [255; 4],
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5,
            thin_film_ior: 1.4,
            geometry_opacity: 1.0,
            geometry_thin_walled: 0,
            // geometry_normal: Default::default(),
            // geometry_tangent: Default::default(),
            // geometry_coat_normal: Default::default(),
            // geometry_coat_tangent: Default::default(),
        }
    }
}

impl Material {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawableMesh {
    pub mesh: String,
    pub transform: Mat4,
    pub material: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub drawables: Vec<DrawableMesh>,
}
