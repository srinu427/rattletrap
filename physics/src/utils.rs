pub fn point_vec4(p: glam::Vec3) -> glam::Vec4 {
    glam::Vec4::from((p, 1.0))
}

pub fn dir_vec4(p: glam::Vec3) -> glam::Vec4 {
    glam::Vec4::from((p, 0.0))
}

pub fn orient_plane(p: glam::Vec4, tr: glam::Mat4) -> glam::Vec4 {
    todo!()
}
