#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Entity(u64);

impl Entity {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

pub struct ComponentData<T> {
    entity_ids: Vec<Entity>,
    data: Vec<T>,
}
