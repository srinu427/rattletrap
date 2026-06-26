use physics::collision_shape::CollisionShape;

pub enum DrawableDisk {}

pub struct GameObjectDisk {
    physics_shape: Option<CollisionShape>,
    drawable: Option<Drawable>,
}
