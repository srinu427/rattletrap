use bytemuck::NoUninit;
use glam::Vec4Swizzles;
use serde::{Deserialize, Serialize};

pub mod vk12;

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, NoUninit)]
pub struct MeshVertex {
    pub pos: glam::Vec3,
    pub norm: glam::Vec3,
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
        let norm = x.cross(y).normalize();
        let vertices = [c + x + y, c - x + y, c - x - y, c + x - y]
            .into_iter()
            .map(|pos| MeshVertex { pos, norm })
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, bytemuck::NoUninit, bytemuck::Zeroable)]
#[serde(default)]
pub struct Material {
    // --- 16-byte Aligned Fields (Vec4 / Color4) ---
    pub base_color: glam::Vec4,
    pub specular_color: glam::Vec4,
    pub transmission_color: glam::Vec4,
    pub subsurface_color: glam::Vec4,
    pub coat_color: glam::Vec4,
    pub fuzz_color: glam::Vec4,
    pub emission_color: glam::Vec4,

    // --- 16-byte Aligned Groups (Vec3 + f32 scalar filler) ---
    pub transmission_scatter: glam::Vec3,
    pub transmission_scatter_anisotropy: f32, // Fills 4-byte padding after Vec3

    pub subsurface_radius_scale: glam::Vec3,
    pub subsurface_scatter_anisotropy: f32, // Fills 4-byte padding after Vec3

    // --- 4-byte Scalar Fields (f32 / u32) ---
    pub base_weight: f32,
    pub base_metalness: f32,
    pub base_diffuse_roughness: f32,
    pub specular_weight: f32,

    pub specular_roughness: f32,
    pub specular_roughness_anisotropy: f32,
    pub specular_ior: f32,
    pub transmission_weight: f32,

    pub transmission_depth: f32,
    pub transmission_dispersion_scale: f32,
    pub transmission_dispersion_abbe_number: f32,
    pub subsurface_weight: f32,

    pub subsurface_radius: f32,
    pub coat_weight: f32,
    pub coat_roughness: f32,
    pub coat_roughness_anisotropy: f32,

    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub fuzz_weight: f32,
    pub fuzz_roughness: f32,

    pub emission_luminance: f32,
    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,

    pub geometry_opacity: f32,
    pub geometry_thin_walled: u32,
    pub _pad0: f32, // Padding to bring total struct size to a multiple of 16 bytes
    pub _pad1: f32,
}

impl Material {
    /// Creates a material with default values adhering to the OpenPBR specification.
    pub fn open_pbr() -> Self {
        Self {
            // Base
            base_color: glam::Vec4::new(0.8, 0.8, 0.8, 1.0),
            base_weight: 1.0,
            base_metalness: 0.0,
            base_diffuse_roughness: 0.0,

            // Specular
            specular_color: glam::Vec4::ONE,
            specular_weight: 1.0,
            specular_roughness: 0.3,
            specular_roughness_anisotropy: 0.0,
            specular_ior: 1.5,

            // Transmission
            transmission_color: glam::Vec4::ONE,
            transmission_weight: 0.0,
            transmission_depth: 0.0,
            transmission_scatter: glam::Vec3::ZERO,
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_scale: 0.0,
            transmission_dispersion_abbe_number: 20.0,

            // Subsurface
            subsurface_color: glam::Vec4::ONE,
            subsurface_weight: 0.0,
            subsurface_radius: 1.0,
            subsurface_radius_scale: glam::Vec3::ONE,
            subsurface_scatter_anisotropy: 0.0,

            // Coat
            coat_color: glam::Vec4::ONE,
            coat_weight: 0.0,
            coat_roughness: 0.0,
            coat_roughness_anisotropy: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,

            // Fuzz
            fuzz_color: glam::Vec4::ONE,
            fuzz_weight: 0.0,
            fuzz_roughness: 0.5,

            // Emission
            emission_color: glam::Vec4::ONE,
            emission_luminance: 0.0,

            // Thin Film
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5, // 0.5 micrometers (500nm)
            thin_film_ior: 1.5,

            // Geometry
            geometry_opacity: 1.0,
            geometry_thin_walled: 0,

            // Padding
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::open_pbr()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawableMesh {
    pub mesh: String,
    pub transform: glam::Mat4,
    pub material: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub drawables: Vec<DrawableMesh>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Camera3d {
    pub eye: glam::Vec3,
    pub dir: glam::Vec3,
    pub up: glam::Vec3,
    pub fov: f32,
    pub aspect: f32,
    pub np: f32,
    pub fp: f32,
}

impl Camera3d {
    pub fn new(
        eye: glam::Vec3,
        dir: glam::Vec3,
        up: glam::Vec3,
        fov: f32,
        aspect: f32,
        np: f32,
        fp: f32,
    ) -> Self {
        let dir = dir.normalize();
        Self {
            eye,
            dir,
            up,
            fov,
            aspect,
            np,
            fp,
        }
    }

    fn get_perspective_proj(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_to_rh(self.eye, self.dir, self.up);
        let proj = glam::Mat4::perspective_rh(self.fov, self.aspect, self.np, self.fp);
        proj * view
    }

    pub fn move_left_right(&mut self, up: glam::Vec3, angle: f32) {
        let rot = glam::Mat4::from_axis_angle(up, angle);
        self.dir = (rot * glam::Vec4::from((self.dir, 0.0))).xyz();
        self.up = (rot * glam::Vec4::from((self.up, 0.0))).xyz();
    }

    pub fn move_up_down(&mut self, up: glam::Vec3, angle: f32) {
        let rot_axis = up.cross(self.dir);
        if rot_axis.length_squared() == 0.0 {
            return;
        }
        let rot_axis = rot_axis.normalize();
        let rot = glam::Mat4::from_axis_angle(rot_axis, angle);
        self.dir = (rot * glam::Vec4::from((self.dir, 0.0))).xyz();
        self.up = (rot * glam::Vec4::from((self.up, 0.0))).xyz();
    }
}
