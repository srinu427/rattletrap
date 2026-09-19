use std::{
    any::{Any, TypeId},
    fs,
    sync::Arc,
};

// use physics::PhysicsManager;
use crate::inputs::Inputs;

use glam::{Mat4, Vec3};
use hashbrown::HashMap;
use physics::{
    Kinematics, Orientation, PhysicsManager, RigidBody, collision_shape::CollisionShape,
};
use renderers::{Camera3d, DrawableMesh, Mesh, Scene, vk12::RendererVk12};
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

#[derive(Debug, Clone, Copy)]
pub struct GameObjectRef {
    renderer_id: i64,
    physics_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameWorldDisk {
    objects: Vec<GameObjectDisk>,
}

pub struct ComponentData<T> {
    data: Vec<T>,
}

pub struct DataBank {
    bank: HashMap<TypeId, Box<dyn Any>>,
}

impl DataBank {
    unsafe fn get_unchecked_mut<T: 'static>(&mut self) -> &mut ComponentData<T> {
        // Get the Box<dyn Any> from the map
        let any_box = self
            .bank
            .get_mut(&TypeId::of::<T>())
            .expect("Component data not found");

        // Convert the Box<dyn Any> to a raw pointer and cast it to the concrete type pointer
        let raw: *mut dyn Any = &mut **any_box;
        let typed_ptr = raw.cast::<ComponentData<T>>();

        &mut *typed_ptr
    }

    unsafe fn get_unchecked<T: 'static>(&self) -> &ComponentData<T> {
        // Get the Box<dyn Any> from the map
        let any_box = self
            .bank
            .get(&TypeId::of::<T>())
            .expect("Component data not found");

        // Convert the Box<dyn Any> to a raw pointer and cast it to the concrete type pointer
        let raw: *const dyn Any = &**any_box;
        let typed_ptr = raw.cast::<ComponentData<T>>();

        &*typed_ptr
    }

    fn get_data<T: 'static>(&self, idx: i64) -> Option<&T> {
        if idx < 0 {
            return None;
        }
        unsafe { self.get_unchecked().data.get(idx as usize) }
    }
}

pub struct GameWorld {
    renderer_scene: Scene,
    physics_sim: Vec<RigidBody>,
    object_refs: Vec<GameObjectRef>,
}

pub struct Game {
    pub(crate) renderer_system: RendererVk12,
    camera: Camera3d,
    physics_system: PhysicsManager,
    world: GameWorld,
    // camera: Cam3d,
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
            renderer_system,
            camera,
            physics_system,
            world: GameWorld {
                renderer_scene: Scene { drawables: vec![] },
                physics_sim: vec![],
                object_refs: vec![],
            },
            // camera, d
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
        let mut drawables = vec![];
        let mut physics_rbs = vec![];
        let mut game_object_refs = vec![];
        for obj in level.objects {
            let mut game_object = GameObjectRef {
                renderer_id: -1,
                physics_id: -1,
            };
            if let Some(drawable) = &obj.renderable {
                if !fs::exists(&drawable.mesh).unwrap_or(false) {
                    if let Some(physics_rb) = &obj.physics {
                        let mesh = Self::shape_to_mesh(&physics_rb.shape);
                        fs::write(&drawable.mesh, ron::ser::to_string(&mesh)?)?;
                        game_object.renderer_id = drawables.len() as _;
                        drawables.push(DrawableMesh {
                            mesh: drawable.mesh.clone(),
                            transform: drawable.transform,
                            material: drawable.material.clone(),
                        });
                    } else {
                        log::error!("cant find mesh {:?}. skipping loading it", &drawable.mesh);
                    }
                } else {
                    game_object.renderer_id = drawables.len() as _;
                    drawables.push(DrawableMesh {
                        mesh: drawable.mesh.clone(),
                        transform: drawable.transform,
                        material: drawable.material.clone(),
                    });
                }
            }
            if let Some(physics_rb) = &obj.physics {
                let rb = Self::import_rb(physics_rb, obj.init_location);
                game_object.physics_id = physics_rbs.len() as _;
                physics_rbs.push(rb);
            }
            game_object_refs.push(game_object);
        }
        self.world = GameWorld {
            renderer_scene: Scene { drawables },
            physics_sim: physics_rbs,
            object_refs: game_object_refs,
        };
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
            self.renderer_system.reset_data();
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
            for rb in self.world.physics_sim.iter_mut() {
                if rb.has_gravity {
                    rb.kinematics.acceleration.y = -10.0;
                }
            }
            self.physics_system.run_ms(&mut self.world.physics_sim);
        }
        for gor in &self.world.object_refs {
            if gor.physics_id < 0 || gor.renderer_id < 0 {
                continue;
            }
            self.world.renderer_scene.drawables[gor.renderer_id as usize].transform =
                self.world.physics_sim[gor.physics_id as usize]
                    .orient
                    .to_transform();
        }
        self.renderer_system
            .render(&self.world.renderer_scene, &self.camera)?;
        inputs.advance_frame();
        Ok(())
    }
}
