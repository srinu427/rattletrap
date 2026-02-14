use glam::Vec4Swizzles;

pub fn point_vec4(p: glam::Vec3) -> glam::Vec4 {
    glam::Vec4::from((p, 1.0))
}

pub fn dir_vec4(p: glam::Vec3) -> glam::Vec4 {
    glam::Vec4::from((p, 0.0))
}

pub fn orient_plane(pl: glam::Vec4, tr: &glam::Mat4) -> glam::Vec4 {
    let new_n = tr * dir_vec4(pl.xyz());
    let new_p = tr * point_vec4(pl.xyz() * pl.w);
    let new_pl = glam::Vec4::from((new_n.xyz(), -new_p.xyz().dot(new_n.xyz())));
    new_pl
}

pub fn new_plane(n: glam::Vec3, p: glam::Vec3) -> glam::Vec4 {
    glam::Vec4::from((n, -n.dot(p)))
}

pub fn get_triangle_normal(a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> glam::Vec3 {
    let e1 = b - a;
    let e2 = c - b;
    e1.cross(e2)
}

pub fn get_triangle_plane(a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> glam::Vec4 {
    let n = get_triangle_normal(a, b, c);
    new_plane(n, a)
}

pub fn points_on_side(pl: glam::Vec4, points: &[glam::Vec4]) -> Option<bool> {
    let mut side = None;
    for p in points {
        let f_p_dist = p.dot(pl);
        match side.as_ref() {
            Some(&pos) => {
                if pos {
                    if f_p_dist < 0.0 {
                        return None;
                    }
                } else {
                    if f_p_dist > 0.0 {
                        return None;
                    }
                }
            }
            None => {
                if f_p_dist == 0.0 {
                    continue;
                } else if f_p_dist > 0.0 {
                    side = Some(true);
                } else {
                    side = Some(false)
                }
            }
        }
    }
    side
}

pub fn points_on_pos(pl: glam::Vec4, points: &[glam::Vec4]) -> bool {
    for p in points {
        if p.dot(pl) < 0.0 {
            return false;
        }
    }
    true
}

pub fn points_min_dist(pl: glam::Vec4, points: &[glam::Vec4]) -> f32 {
    let mut min_dist = f32::INFINITY;
    for p in points {
        let f_p_dist = p.dot(pl);
        if f_p_dist < min_dist {
            min_dist = f_p_dist;
        }
    }
    min_dist
}

pub fn points_min_max_dist(pl: glam::Vec4, points: &[glam::Vec4]) -> (f32, f32) {
    let mut min_dist = f32::INFINITY;
    let mut max_dist = f32::NEG_INFINITY;
    for p in points {
        let f_p_dist = p.dot(pl);
        if f_p_dist > max_dist {
            max_dist = f_p_dist;
        }
        if f_p_dist < min_dist {
            min_dist = f_p_dist;
        }
    }
    (min_dist, max_dist)
}

pub fn remove_component(v: &mut glam::Vec3, d: glam::Vec3) {
    *v = *v - (d.dot(*v) * d);
}
