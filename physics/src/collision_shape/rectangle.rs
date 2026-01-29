use glam::Vec4Swizzles;

#[derive(Debug, Clone)]
pub struct Rectangle {
    pub(crate) c: glam::Vec4,
    pub(crate) pl: glam::Vec4,
    pub(crate) u: glam::Vec4,
    pub(crate) v: glam::Vec4,
    pub(crate) points: [glam::Vec4; 4],
    pub(crate) edge_planes: [glam::Vec4; 4],
}

impl Rectangle {
    pub fn new(c: glam::Vec3, u: glam::Vec3, v: glam::Vec3) -> Self {
        let n = u.cross(v).normalize();
        let points = [
            glam::Vec4::from((c + u + v, 1.0)),
            glam::Vec4::from((c + u - v, 1.0)),
            glam::Vec4::from((c - u + v, 1.0)),
            glam::Vec4::from((c - u - v, 1.0)),
        ];
        let edges = [
            points[1] - points[0],
            points[2] - points[1],
            points[3] - points[2],
            points[0] - points[3],
        ];
        let mut edge_planes = [
            glam::Vec4::from((edges[0].xyz().cross(n).normalize(), 0.0)),
            glam::Vec4::from((edges[1].xyz().cross(n).normalize(), 0.0)),
            glam::Vec4::from((edges[2].xyz().cross(n).normalize(), 0.0)),
            glam::Vec4::from((edges[3].xyz().cross(n).normalize(), 0.0)),
        ];
        for i in 0..4 {
            edge_planes[i].z = edge_planes[i].xyz().dot(points[i].xyz())
        }

        Self {
            c: glam::Vec4::from((c, 1.0)),
            pl: glam::Vec4::from((n, n.dot(c))),
            u: glam::Vec4::from((u, 1.0)),
            v: glam::Vec4::from((v, 1.0)),
            points,
            edge_planes,
        }
    }

    pub fn with_orientation(&self, trans: glam::Vec3, rot: glam::Mat4) -> Self {
        let out_transform =
            glam::Mat4::from_translation(trans) * rot * glam::Mat4::from_translation(-self.c.xyz());
        let mut out = self.clone();
        out.c = out_transform * out.c;
        out.pl = out_transform * glam::Vec4::from((out.pl.xyz(), 0.0));
        out.pl.w = out.pl.xyz().dot(out.c.xyz());
        out.u = out_transform * out.u;
        out.v = out_transform * out.v;
        for point in &mut out.points {
            *point = out_transform * *point;
        }
        for ep in &mut out.edge_planes {
            *ep = out_transform * *ep;
        }
        return out;
    }
}
