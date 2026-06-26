pub enum DiskPhysicsShape {
    Sphere {
        c: [f32; 3],
        r: f32,
    },
    Rect {
        c: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
    },
    Cube {
        c: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        h: f32,
    },
}

pub struct DiskPhysicsInfo {
    shape: DiskPhysicsShape,
    gravity: bool,
}

pub enum DiskRenderInfo {
    TexturedMesh { mesh: String, tex: String },
}

pub struct DiskNode {
    name: String,
    physics_info: Option<DiskPhysicsInfo>,
    render_info: Option<DiskRenderInfo>,
    children: Vec<Self>,
}
