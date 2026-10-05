use std::{fs, sync::Arc};

// use physics::PhysicsManager;
use crate::inputs::Inputs;

use ecs::{EcsData, Entity};
use glam::{Mat4, Vec3};
use physics::{
    Kinematics, Orientation, PhysicsManager, RigidBody, collision_shape::CollisionShape,
};
use renderers::{Camera3d, DrawableMesh, Light, Mesh, renderer::RendererVk12};
use serde::{Deserialize, Serialize};
use winit::{
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderableDisk {
    pub mesh: String,
    #[serde(default = "default_transform")]
    pub transform: Mat4,
    pub material: String,
}

fn default_transform() -> glam::Mat4 {
    glam::Mat4::IDENTITY
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapeDisk {
    Rectangle {
        c: [f32; 3],
        x: [f32; 3],
        y: [f32; 3],
    },
    Cube {
        c: [f32; 3],
        x: [f32; 3],
        y: [f32; 3],
        h: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsDisk {
    pub mass: f32,
    pub shape: ShapeDisk,
    pub has_gravity: bool,
    pub no_interact_mask: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameObjectDisk {
    renderable: Option<RenderableDisk>,
    physics: Option<PhysicsDisk>,
    init_location: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameWorldDisk {
    objects: Vec<GameObjectDisk>,
}

pub struct Game {
    pub ecs_data: EcsData,
    pub(crate) renderer_system: RendererVk12,
    camera: Camera3d,
    physics_system: PhysicsManager,
    entities: Vec<Entity>,
    window: Arc<Window>,
    is_cursor_grabbed: bool,
}

impl Game {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let renderer_system = RendererVk12::new(&window)?;
        let physics_system = PhysicsManager::new();

        let camera = Camera3d::new(
            glam::vec3(3.0, 3.0, 3.0),
            glam::vec3(-1.0, -1.0, -1.0),
            glam::Vec3::Y,
            2.0,
            1.0,
            0.1,
            100.0,
        );

        Ok(Self {
            ecs_data: EcsData::new(),
            renderer_system,
            camera,
            physics_system,
            entities: Default::default(),
            window,
            is_cursor_grabbed: true,
        })
    }

    fn toggle_mouse_grab(&mut self) {
        if self.is_cursor_grabbed {
            if let Err(e) = self.window.set_cursor_grab(CursorGrabMode::None) {
                log::warn!("releasing cursor grab failed: {e}");
                return;
            }
            self.is_cursor_grabbed = false;
        } else {
            if let Err(e) = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked))
            {
                log::warn!("cursor grabbing failed: {e}");
                return;
            }
            self.is_cursor_grabbed = true;
        };
    }

    fn shape_to_mesh(shape: &ShapeDisk) -> Mesh {
        match shape {
            ShapeDisk::Rectangle { c, x, y } => Mesh::new_rectangle(
                glam::Vec3::from(*c),
                glam::Vec3::from(*x),
                glam::Vec3::from(*y),
            ),
            ShapeDisk::Cube { c, x, y, h } => Mesh::new_cube(
                glam::Vec3::from(*c),
                glam::Vec3::from(*x),
                glam::Vec3::from(*y),
                *h,
            ),
        }
    }

    fn import_rb(rb: &PhysicsDisk, location: [f32; 3]) -> RigidBody {
        let shape = match &rb.shape {
            ShapeDisk::Rectangle { c, x, y } => CollisionShape::new_rect(
                Vec3::from_array(*c),
                Vec3::from_array(*x),
                Vec3::from_array(*y),
            ),
            ShapeDisk::Cube { c, x, y, h } => CollisionShape::new_cube(
                Vec3::from_array(*c),
                Vec3::from_array(*x),
                Vec3::from_array(*y),
                *h,
            ),
        };
        let orient = Orientation {
            translation: Vec3::from_array(location),
            rotation: Mat4::IDENTITY,
        };
        RigidBody::new(
            rb.mass,
            Arc::new(shape),
            orient,
            Kinematics::new(),
            false,
            rb.has_gravity,
            rb.no_interact_mask,
        )
    }

    pub fn load_level(&mut self) -> anyhow::Result<()> {
        let level: GameWorldDisk = ron::de::from_bytes(&fs::read("data/levels/2.ron")?)?;
        for ent in self.entities.drain(..) {
            self.ecs_data.remove_entity(ent);
        }
        for obj in level.objects {
            let new_ent = self.ecs_data.new_entity();
            self.entities.push(new_ent);
            if let Some(drawable) = &obj.renderable {
                if !fs::exists(&drawable.mesh).unwrap_or(false) {
                    if let Some(physics_rb) = &obj.physics {
                        let mesh = Self::shape_to_mesh(&physics_rb.shape);
                        fs::write(&drawable.mesh, ron::ser::to_string(&mesh)?)?;
                        self.ecs_data.insert_component(
                            new_ent,
                            DrawableMesh {
                                mesh: drawable.mesh.clone(),
                                transform: drawable.transform,
                                material: drawable.material.clone(),
                            },
                        );
                    } else {
                        log::error!("cant find mesh {:?}. skipping loading it", &drawable.mesh);
                    }
                } else {
                    self.ecs_data.insert_component(
                        new_ent,
                        DrawableMesh {
                            mesh: drawable.mesh.clone(),
                            transform: drawable.transform,
                            material: drawable.material.clone(),
                        },
                    );
                }
            }
            if let Some(physics_rb) = &obj.physics {
                let rb = Self::import_rb(physics_rb, obj.init_location);
                self.ecs_data.insert_component(new_ent, rb);
            }
        }
        // hardcode some lights
        let lights_count = self
            .ecs_data
            .comp_data_iter::<Light>()
            .map(|it| it.count())
            .unwrap_or(0);
        if lights_count == 0 {
            let light1 = self.ecs_data.new_entity();
            self.ecs_data.insert_component(
                light1,
                Light::new_directional_light(
                    glam::vec3(1.0, 1.0, 1.0),
                    4.0,
                    glam::vec3(-1.0, -0.9, -0.8),
                ),
            );
            let light2 = self.ecs_data.new_entity();
            self.ecs_data.insert_component(
                light2,
                Light::new_point_light(
                    glam::vec3(0.0, 6.0, 0.0),
                    glam::vec3(1.0, 0.5, 0.0),
                    1000.0,
                    20.0,
                ),
            );
        }
        self.renderer_system.reload_resources(&mut self.ecs_data)?;
        Ok(())
    }

    fn camera_move(&mut self, frame_time: u128, front: i32, right: i32, up: i32) {
        let mvmt = 0.002 * (frame_time as f32);
        self.camera.eye.y += up as f32 * mvmt;

        let mut dir_proj = self.camera.dir;
        dir_proj.y = 0.0;
        if dir_proj.x == 0.0 && dir_proj.z == 0.0 {
            dir_proj = -self.camera.up;
            dir_proj.y = 0.0;
        }
        let x = dir_proj.normalize();
        let y = glam::vec3(-x.z, 0.0, x.x);
        self.camera.eye += front as f32 * x * mvmt;
        self.camera.eye += right as f32 * y * mvmt;
    }

    pub fn run(&mut self, frame_time: u128, inputs: &mut Inputs) -> anyhow::Result<()> {
        let mouse_move = inputs.mouse_delta();
        if inputs.key_pressed_this_frame(PhysicalKey::Code(KeyCode::KeyG)) {
            self.toggle_mouse_grab();
        }
        if inputs.key_pressed_this_frame(PhysicalKey::Code(KeyCode::KeyR)) {
            println!("refreshing level");
            self.load_level()
                .inspect_err(|e| log::error!("loading level failed: {e:#}"))
                .ok();
        }
        let mut up = 0;
        let mut front = 0;
        let mut right = 0;
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::Space)) {
            up += 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyC)) {
            up -= 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyW)) {
            front += 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyS)) {
            front -= 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyD)) {
            right += 1;
        }
        if inputs.key_pressed(PhysicalKey::Code(KeyCode::KeyA)) {
            right -= 1;
        }
        self.camera_move(frame_time, front, right, up);
        if self.is_cursor_grabbed {
            self.camera
                .move_left_right(glam::Vec3::Y, -0.01 * mouse_move.0 as f32);
            self.camera
                .move_up_down(glam::Vec3::Y, 0.01 * mouse_move.1 as f32);
        }
        for _ in 0..frame_time {
            let Some(rb_data) = self.ecs_data.comp_data_iter_mut::<RigidBody>() else {
                continue;
            };
            for (_, rb) in rb_data {
                if rb.has_gravity {
                    rb.kinematics.acceleration.y = -10.0;
                }
            }
            self.physics_system.run_ms(&mut self.ecs_data);
        }

        let mut phy_transforms = vec![];
        if let Some(rb_iter) = self.ecs_data.comp_data_iter::<RigidBody>() {
            for (ent, rb) in rb_iter {
                phy_transforms.push((ent, rb.orient.to_transform()));
            }
        }
        for (ent, transform) in phy_transforms {
            let Some(dm) = self.ecs_data.get_component_mut::<DrawableMesh>(ent) else {
                continue;
            };
            dm.transform = transform;
        }
        self.renderer_system
            .render(&mut self.ecs_data, &self.camera)?;
        inputs.advance_frame();
        Ok(())
    }
}
