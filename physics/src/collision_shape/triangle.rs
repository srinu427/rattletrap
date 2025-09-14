#[derive(Debug, Clone, getset::CopyGetters, getset::Getters)]
pub struct Triangle {
    #[getset(get = "pub")]
    points: [glam::Vec3; 3],
    #[getset(get = "pub")]
    side_len: [f32; 3],
    #[getset(get_copy = "pub")]
    normal: glam::Vec3,
    #[getset(get = "pub")]
    bound_planes: [glam::Vec4; 3],
    #[getset(get_copy = "pub")]
    radius: f32,
}

impl Triangle {
    pub fn new(points: [glam::Vec3; 3], radius: f32) -> Self {
        let side_lines = [
            points[1] - points[0],
            points[2] - points[1],
            points[0] - points[2],
        ];

        let side_len = side_lines.map(|s| s.length());

        let normal = side_lines[0].cross(side_lines[1]).normalize();

        let bp_normals = side_lines.map(|line| line.cross(normal).normalize());

        let mut bound_planes = [glam::Vec4::ZERO; 3];

        for i in 0..3 {
            bound_planes[i] = glam::Vec4::from((bp_normals[i], -bp_normals[i].dot(points[i])));
        }

        Self {
            points,
            side_len,
            normal,
            bound_planes,
            radius,
        }
    }
}