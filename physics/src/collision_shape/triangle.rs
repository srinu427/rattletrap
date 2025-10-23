use glam::Vec4Swizzles;

#[derive(Debug, Clone, getset::CopyGetters, getset::Getters)]
pub struct Triangle {
    pub(crate) points: [glam::Vec3; 3],
    pub(crate) side_len: [f32; 3],
    pub(crate) normal: glam::Vec3,
    pub(crate) bound_planes: [glam::Vec4; 3],
    pub(crate) radius: f32,
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

    pub fn apply_orientation(&self, trans: glam::Vec3, rot: glam::Mat4) -> Self {
        let new_points = self
            .points
            .map(|p| (rot * glam::Vec4::from((p, 1.0))).xyz() + trans);
        let new_normal = (rot * glam::Vec4::from((self.normal, 1.0))).xyz();
        let new_bound_planes = self.bound_planes.map(|bp| {
            let n = bp.xyz();
            let new_n = (rot * glam::Vec4::from((n, 0.0))).xyz();
            let new_d = bp.w + new_n.dot(trans);
            glam::Vec4::from((new_n, new_d))
        });

        Self {
            points: new_points,
            side_len: self.side_len.clone(),
            normal: new_normal,
            bound_planes: new_bound_planes,
            radius: self.radius,
        }
    }
}
