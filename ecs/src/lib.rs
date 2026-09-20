use std::{
    any::{Any, TypeId},
    mem,
};

use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Entity(u64);

struct ComponentData<T> {
    data: Vec<(Entity, T)>,
}

pub struct EcsData {
    last_ent: u64,
    bank: HashMap<TypeId, Box<dyn Any>>,
    entity_idxs: HashMap<Entity, HashMap<TypeId, usize>>,
}

impl EcsData {
    fn init_comp_data<T: 'static>(&mut self) {
        let type_id = TypeId::of::<T>();
        if !self.bank.contains_key(&type_id) {
            self.bank
                .insert(type_id, Box::new(ComponentData::<T> { data: vec![] }));
        }
    }

    unsafe fn get_comp_data_unchecked_mut<T: 'static>(&mut self) -> Option<&mut ComponentData<T>> {
        // Get the Box<dyn Any> from the map
        let any_box = self.bank.get_mut(&TypeId::of::<T>())?;

        // Convert the Box<dyn Any> to a raw pointer and cast it to the concrete type pointer
        let raw: *mut dyn Any = &mut **any_box;
        let typed_ptr = raw.cast::<ComponentData<T>>();

        Some(unsafe { &mut *typed_ptr })
    }

    unsafe fn get_comp_data_unchecked<T: 'static>(&self) -> Option<&ComponentData<T>> {
        // Get the Box<dyn Any> from the map
        let any_box = self.bank.get(&TypeId::of::<T>())?;

        // Convert the Box<dyn Any> to a raw pointer and cast it to the concrete type pointer
        let raw: *const dyn Any = &**any_box;
        let typed_ptr = raw.cast::<ComponentData<T>>();

        Some(unsafe { &*typed_ptr })
    }

    pub fn new_entity(&mut self) -> Entity {
        self.last_ent += 1;
        let new_ent = Entity(self.last_ent);
        self.entity_idxs.insert(new_ent, HashMap::new());
        new_ent
    }

    fn swap_remove_update(&mut self, type_id: TypeId, idx: usize) {}

    pub fn remove_entity(&mut self, entity: Entity) {
        let Some(idxs) = self.entity_idxs.remove(&entity) else {
            return;
        };
        for (type_id, idx) in idxs {}
    }

    pub fn insert_component<T: 'static>(&mut self, entity: Entity, mut elem: T) -> Option<T> {
        let type_id = TypeId::of::<T>();
        if !self.bank.contains_key(&type_id) {
            self.bank
                .insert(type_id, Box::new(ComponentData::<T> { data: vec![] }));
        }

        let old_idx = match self.entity_idxs.get(&entity) {
            Some(h) => h.get(&type_id).cloned(),
            None => None,
        };
        match old_idx {
            Some(old_idx) => {
                let data_obj = match unsafe { self.get_comp_data_unchecked_mut::<T>() } {
                    Some(t) => t,
                    None => unreachable!(),
                };
                mem::swap(&mut data_obj.data[old_idx].1, &mut elem);
                Some(elem)
            }
            None => {
                let data_obj = match unsafe { self.get_comp_data_unchecked_mut::<T>() } {
                    Some(t) => t,
                    None => unreachable!(),
                };
                let new_idx = data_obj.data.len();
                data_obj.data.push((entity, elem));
                self.entity_idxs
                    .entry(entity)
                    .or_insert(Default::default())
                    .insert(type_id, new_idx);
                None
            }
        }
    }
}
