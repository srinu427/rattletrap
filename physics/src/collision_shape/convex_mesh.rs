use glam::Vec4Swizzles;

use crate::utils::{get_triangle_plane, new_plane, orient_plane};

#[derive(Debug, Clone)]
pub struct ConvexMesh {
    pub(crate) points: Vec<glam::Vec4>,
    pub(crate) edges: Vec<(usize, usize)>,
    pub(crate) face_points: Vec<Vec<usize>>,
    pub(crate) faces: Vec<glam::Vec4>,
    pub(crate) face_bounds: Vec<Vec<glam::Vec4>>,
}

impl ConvexMesh {
    fn calc_fbs(n: glam::Vec3, points: &[glam::Vec4], face_points: &[usize]) -> Vec<glam::Vec4> {
        let mut edge_planes = Vec::with_capacity(points.len());
        for i in 0..face_points.len() {
            let j = (i + 1) % face_points.len();
            let a = points[i];
            let b = points[j];
            let edge = b - a;
            let edge_n = edge.xyz().cross(n).normalize();
            edge_planes.push(new_plane(edge_n, a.xyz()));
        }
        edge_planes
    }

    pub fn from_points_edges_faces(
        points: Vec<glam::Vec4>,
        edges: Vec<(usize, usize)>,
        face_points: Vec<Vec<usize>>,
    ) -> Self {
        let faces: Vec<_> = face_points
            .iter()
            .map(|fps| {
                get_triangle_plane(
                    points[fps[0]].xyz(),
                    points[fps[1]].xyz(),
                    points[fps[2]].xyz(),
                )
            })
            .collect();

        let face_bounds: Vec<_> = face_points
            .iter()
            .zip(faces.iter())
            .map(|(fps, face)| Self::calc_fbs(face.xyz(), &points, fps))
            .collect();

        Self {
            points,
            edges,
            face_points,
            faces,
            face_bounds,
        }
    }

    pub fn new_rect(c: glam::Vec3, u: glam::Vec3, v: glam::Vec3) -> Self {
        let points = vec![
            glam::Vec4::from((c + u + v, 1.0)),
            glam::Vec4::from((c + u - v, 1.0)),
            glam::Vec4::from((c - u + v, 1.0)),
            glam::Vec4::from((c - u - v, 1.0)),
        ];
        let edges = vec![(0, 1), (2, 1), (3, 2), (0, 3)];
        let faces = vec![vec![0, 1, 2, 3]];
        Self::from_points_edges_faces(points, edges, faces)
    }

    pub fn with_orientation(&self, trans: glam::Vec3, rot: glam::Mat4) -> Self {
        let out_transform = glam::Mat4::from_translation(trans) * rot;
        let new_points: Vec<_> = self.points.iter().map(|p| out_transform * p).collect();
        let new_faces: Vec<_> = self
            .faces
            .iter()
            .map(|&pl| orient_plane(pl, &out_transform))
            .collect();
        let new_face_bounds: Vec<Vec<_>> = self
            .face_bounds
            .iter()
            .map(|fbs| {
                fbs.iter()
                    .map(|&pl| orient_plane(pl, &out_transform))
                    .collect()
            })
            .collect();
        Self {
            points: new_points,
            edges: self.edges.clone(),
            face_points: self.face_points.clone(),
            faces: new_faces,
            face_bounds: new_face_bounds,
        }
    }
}
