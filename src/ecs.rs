#[derive(Debug)]
pub struct Node {
    name: String,
    entity_id: usize,
    children: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct Entity {
    renderer_info: Option<RenderInfo>,
    physics_info: Option<PhysicsInfo>,
}

#[derive(Debug, Clone)]
pub struct RenderInfo {
    mesh_id: usize,
    texture_id: usize,
}

#[derive(Debug, Clone)]
pub struct PhysicsInfo {
    obj_id: usize,
    pos: glam::Vec3,
    rot: glam::Mat4,
    full_t: glam::Mat4,
    vel: glam::Vec3,
    acc: glam::Vec3,
}

#[derive(Debug)]
pub struct EcsMega {
    scene_root: Node,
    entities: Vec<Entity>,
}
