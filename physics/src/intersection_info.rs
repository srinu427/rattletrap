use glam::{Vec3, Vec4Swizzles};

use crate::{collision_shape::CollisionShape, utils::get_triangle_plane};

const EPA_PROGRESS_EPSILON: f32 = 0.0001;

#[derive(Debug, Clone)]
pub struct IntersectionInfo {
    pub dir: Vec3,
    pub point_1: Vec3,
    pub point_2: Vec3,
    pub dist: f32,
}

impl IntersectionInfo {
    pub fn new(
        a: &Box<dyn CollisionShape>,
        b: &Box<dyn CollisionShape>,
    ) -> Option<IntersectionInfo> {
        let mut supp_points: Vec<Vec3> = vec![];
        loop {
            let check_dir = if supp_points.len() == 0 {
                (b.center_hint() - a.center_hint()).normalize()
            } else if supp_points.len() == 1 {
                -supp_points[0].normalize()
            } else if supp_points.len() == 2 {
                let ab = supp_points[1] - supp_points[0];
                let ao = -supp_points[0];
                let po = ao.reject_from(ab);
                if po.length_squared() == 0.0 {
                    // Origin is on AB
                    let sample = if ab.x >= 0.5 { Vec3::Y } else { Vec3::X };
                    let p_dir = sample.reject_from(ab);
                    p_dir.normalize()
                } else {
                    po.normalize()
                }
            } else if supp_points.len() == 3 {
                let abc = get_triangle_plane(supp_points[0], supp_points[1], supp_points[2]);
                if abc.w >= 0.0 { abc.xyz() } else { -abc.xyz() }
            } else {
                break;
            };
            let farth_point_a = a.farthest_point_along(check_dir);
            let farth_point_b = b.farthest_point_along(-check_dir);
            let supp_point = farth_point_a - farth_point_b;
            supp_points.push(supp_point);
            // Check if new point crossed origin
            if check_dir.dot(supp_point) < 0.0 {
                return None;
            }
            if supp_point == Vec3::ZERO {
                return Some(Self {
                    dir: check_dir,
                    point_1: farth_point_a,
                    point_2: farth_point_b,
                    dist: 0.0,
                });
            }
        }
        // Check if simplex is a tetrahedron
        // If it is, find the max penetration or separation
        if supp_points.len() == 4 {
            let mut points = supp_points.clone();
            let mut pen_dir = Vec3::ZERO;
            let mut pen_point_a = Vec3::ZERO;
            let mut pen_point_b = Vec3::ZERO;
            let mut max_face_dist = f32::NEG_INFINITY;

            loop {
                // Calculate tetrahedron faces
                let mut faces = vec![];
                for i in 0..4 {
                    let mut tri_ids = [i, (i + 1) % 4, (i + 2) % 4];
                    let mut tri_plane = get_triangle_plane(
                        points[tri_ids[0]],
                        points[tri_ids[1]],
                        points[tri_ids[2]],
                    );
                    if tri_plane.w < 0.0 {
                        tri_plane = -tri_plane;
                        tri_ids.swap(0, 1);
                    }
                    faces.push((tri_plane, tri_ids));
                }
                // Find closest face
                let mut c_min_dist_face_idx = 0;
                for (i, face) in faces[1..].iter().enumerate() {
                    if face.0.w < faces[c_min_dist_face_idx].0.w {
                        c_min_dist_face_idx = i;
                    }
                }
                let new_dir = faces[c_min_dist_face_idx].0.xyz();
                let farth_point_a = a.farthest_point_along(new_dir);
                let farth_point_b = b.farthest_point_along(-new_dir);
                let new_point = farth_point_a - farth_point_b;
                let face_dist = faces[c_min_dist_face_idx].0.w;
                if face_dist - EPA_PROGRESS_EPSILON <= max_face_dist {
                    break;
                } else {
                    points = vec![
                        points[faces[c_min_dist_face_idx].1[0]],
                        points[faces[c_min_dist_face_idx].1[1]],
                        points[faces[c_min_dist_face_idx].1[2]],
                        new_point,
                    ];
                    pen_dir = new_dir;
                    pen_point_a = farth_point_a;
                    pen_point_b = farth_point_b;
                    max_face_dist = face_dist;
                }
            }
            return Some(Self {
                dir: pen_dir,
                point_1: pen_point_a,
                point_2: pen_point_b,
                dist: (pen_point_a - pen_point_b).length(),
            });
        } else {
            return None;
        }
    }

    pub fn obj_swapped(mut self) -> Self {
        self.dir = -self.dir;
        (self.point_1, self.point_2) = (self.point_2, self.point_1);
        self
    }
}
