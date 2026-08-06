use glam::{Vec3, Vec4Swizzles};

use crate::{collision_shape::CollisionShape, utils::get_triangle_plane};

const EPA_PROGRESS_EPSILON: f32 = 0.0001;
const MIN_EPA_CYCLES: usize = 4;

#[derive(Debug, Clone)]
pub struct IntersectionInfo {
    pub dir: Vec3,
    pub dist: f32,
}

impl IntersectionInfo {
    pub fn new_with_gjk(a: &CollisionShape, b: &CollisionShape) -> Option<IntersectionInfo> {
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
            // Check if new point crossed origin
            let supp_pt_check_dist = check_dir.dot(supp_point);
            if supp_pt_check_dist < 0.0 {
                return None;
            }
            supp_points.push(supp_point);
            if supp_point == Vec3::ZERO {
                // println!("touch found");
                return Some(Self {
                    dir: check_dir,
                    dist: 0.0,
                });
            }
        }
        // Check if simplex is a tetrahedron
        // If it is, find the max penetration or separation
        if supp_points.len() == 4 {
            let mut points = supp_points.clone();
            let points_len = points.len();
            let mut face_pts: Vec<_> = (0..points_len)
                .map(|i| [i, (i + 1) % points_len, (i + 2) % points_len])
                .collect();
            let mut pen_dir;
            let mut max_face_dist = f32::NEG_INFINITY;
            let mut epa_cycles = 0;
            'epa: loop {
                epa_cycles += 1;
                // Calculate tetrahedron faces
                let mut faces = vec![];
                for idxs in &face_pts {
                    let mut tri_plane =
                        get_triangle_plane(points[idxs[0]], points[idxs[1]], points[idxs[2]]);
                    if tri_plane.w < 0.0 {
                        tri_plane = -tri_plane;
                    }
                    faces.push(tri_plane);
                }
                // Find closest face
                let mut c_min_dist_face_idx = 0;
                for (i, face) in faces.iter().enumerate() {
                    if face.w < faces[c_min_dist_face_idx].w {
                        c_min_dist_face_idx = i;
                    }
                }
                let new_dir = -faces[c_min_dist_face_idx].xyz();
                let farth_point_a = a.farthest_point_along(new_dir);
                let farth_point_b = b.farthest_point_along(-new_dir);
                let new_point = farth_point_a - farth_point_b;
                let face_dist = faces[c_min_dist_face_idx].w;
                if face_dist - EPA_PROGRESS_EPSILON <= max_face_dist && epa_cycles > MIN_EPA_CYCLES
                {
                    pen_dir = new_dir;
                    max_face_dist = face_dist;
                    break 'epa;
                } else {
                    pen_dir = new_dir;
                    max_face_dist = face_dist;
                    for p in &points {
                        if *p == new_point {
                            break 'epa;
                        }
                    }
                    points = vec![
                        points[face_pts[c_min_dist_face_idx][0]],
                        points[face_pts[c_min_dist_face_idx][1]],
                        points[face_pts[c_min_dist_face_idx][2]],
                        new_point,
                    ];
                    face_pts = vec![[0, 1, 3], [1, 2, 3], [0, 2, 3]];
                }
            }
            return Some(Self {
                dir: pen_dir,
                dist: max_face_dist,
            });
        } else {
            return None;
        }
    }

    pub fn obj_swapped(mut self) -> Self {
        self.dir = -self.dir;
        self
    }
}
