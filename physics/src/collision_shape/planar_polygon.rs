use glam::Vec4Swizzles;

#[derive(Debug, Clone)]
pub struct PlanarPolygon {
    pub(crate) pl: glam::Vec4,
    pub(crate) points: Vec<glam::Vec4>,
    pub(crate) edge_planes: Vec<glam::Vec4>,
}

impl PlanarPolygon {
    fn calc_eps(n: glam::Vec3, points: &[glam::Vec4]) -> Vec<glam::Vec4> {
        let mut edge_planes = Vec::with_capacity(points.len());
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            let edge = b - a;
            let edge_n = edge.xyz().cross(n).normalize();
            edge_planes.push(glam::Vec4::from((edge_n, 0.0)));
        }
        edge_planes
    }

    pub fn from_points(points: Vec<glam::Vec4>) -> Self {
        let e1 = points[1] - points[0];
        let e2 = points[2] - points[1];
        let n = e1.xyz().cross(e2.xyz()).normalize();
        let edge_planes = Self::calc_eps(n, &points);
        Self {
            pl: glam::Vec4::from((n, -n.dot(points[0].xyz()))),
            points: points.to_vec(),
            edge_planes,
        }
    }

    pub fn new_rect(c: glam::Vec3, u: glam::Vec3, v: glam::Vec3) -> Self {
        let points = vec![
            glam::Vec4::from((c + u + v, 1.0)),
            glam::Vec4::from((c + u - v, 1.0)),
            glam::Vec4::from((c - u + v, 1.0)),
            glam::Vec4::from((c - u - v, 1.0)),
        ];
        Self::from_points(points)
    }

    pub fn with_orientation(&self, trans: glam::Vec3, rot: glam::Mat4) -> Self {
        let out_transform = glam::Mat4::from_translation(trans) * rot;
        let new_n = out_transform * glam::Vec4::from((self.pl.xyz(), 0.0));
        let new_points: Vec<_> = self.points.iter().map(|p| out_transform * p).collect();
        let new_eps = Self::calc_eps(new_n.xyz(), &new_points);
        Self {
            pl: glam::Vec4::from((new_n.xyz(), -new_n.xyz().dot(new_points[0].xyz()))),
            points: new_points,
            edge_planes: new_eps,
        }
    }
}
