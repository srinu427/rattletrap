use std::sync::Arc;

// use physics::PhysicsManager;
use crate::inputs::Inputs;
use renderer::{MeshDrawInfo, Renderer, camera::Cam3d, mesh::MeshCreateInfo};

use avk12::device::Instance;
use common::Entity;
use indexmap::IndexMap;
use physics::{
    Kinematics, PhysicsManager, RigidBody, collision_shape::CollisionShape, orient::Orientation,
};
use winit::window::Window;
mod game_object;

#[derive(Debug, Clone)]
pub struct PhysicsInfo {
    pos: glam::Vec3,
    rot: glam::Mat4,
    full_t: glam::Mat4,
    vel: glam::Vec3,
    acc: glam::Vec3,
}

pub struct Game {
    pub(crate) renderer_system: Renderer,
    physics_system: PhysicsManager,
    entities: Vec<Entity>,
    mesh_draw_infos: IndexMap<Entity, MeshDrawInfo>,
    camera: Cam3d,
    rigid_bodies: IndexMap<Entity, RigidBody>,
}

impl Game {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let inst = Instance::new(&window)?;
        let device = inst.init_device(0)?;

        let mut renderer_system = Renderer::new(device)?;
        let mut mesh_draw_infos = IndexMap::new();
        let physics_system = PhysicsManager::new();
        let mut rigid_bodies = IndexMap::new();

        // Sample rect object info
        let c = [0., 0., 0.];
        let u = [1., 0., 0.];
        let v = [0., 1., 0.];

        let entity = Entity::new(0);
        let mut mesh_draw = renderer_system.load_mesh_draw(
            "rect1".to_string(),
            MeshCreateInfo::RectCUV { c, u, v },
            "../data/textures/default.png".to_string(),
            true,
        )?;
        mesh_draw.set_transform(glam::Mat4::from_translation(glam::vec3(0., 5., 0.)));
        mesh_draw_infos.insert(entity, mesh_draw);
        rigid_bodies.insert(
            entity,
            RigidBody::new(
                1.0,
                Arc::new(CollisionShape::new_rect(
                    glam::Vec3::from_array(c),
                    glam::Vec3::from_array(u),
                    glam::Vec3::from_array(v),
                )),
                Orientation {
                    translation: glam::vec3(0., 5., 0.),
                    rotation: glam::Mat4::IDENTITY,
                },
                Kinematics::new(),
                true,
                true,
                0,
            ),
        );

        // Another rect object info
        let c = [0.; 3];
        let u = [8., 0., 0.];
        let v = [0., 0., -8.];

        let entity = Entity::new(1);
        let mesh_draw = renderer_system.load_mesh_draw(
            "rect2".to_string(),
            MeshCreateInfo::RectCUV { c, u, v },
            "../data/textures/default.png".to_string(),
            true,
        )?;
        mesh_draw_infos.insert(entity, mesh_draw);
        rigid_bodies.insert(
            entity,
            RigidBody::new(
                f32::INFINITY,
                Arc::new(CollisionShape::new_rect(
                    glam::Vec3::from_array(c),
                    glam::Vec3::from_array(u),
                    glam::Vec3::from_array(v),
                )),
                Orientation::new(),
                Kinematics::new(),
                true,
                false,
                0,
            ),
        );

        let camera = Cam3d::new(
            glam::vec3(3., 3., 3.),
            glam::vec3(-1., -1., -1.),
            glam::Vec3::Y,
            2.,
            1.,
        );

        Ok(Self {
            renderer_system,
            physics_system,
            entities: vec![entity],
            mesh_draw_infos,
            camera,
            rigid_bodies,
        })
    }

    pub fn run(&mut self, frame_time: u128, inputs: &mut Inputs) -> anyhow::Result<()> {
        let mouse_move = inputs.mouse_delta();
        for _ in 0..frame_time {
            for (_, rb) in &mut self.rigid_bodies {
                if rb.has_gravity {
                    rb.kinematics.acceleration.y = -10.0;
                }
            }
            self.physics_system.run_ms(&mut self.rigid_bodies);
        }

        for (ent, rb) in &self.rigid_bodies {
            if let Some(mdi) = self.mesh_draw_infos.get_mut(ent) {
                mdi.set_transform(rb.orient.to_transform());
            }
        }

        self.camera
            .move_left_right(glam::Vec3::Y, -0.01 * mouse_move.0 as f32);
        self.camera
            .move_up_down(glam::Vec3::Y, 0.01 * mouse_move.1 as f32);
        self.renderer_system
            .render(&mut self.camera, &self.mesh_draw_infos)?;
        inputs.advance_frame();
        Ok(())
    }
}
