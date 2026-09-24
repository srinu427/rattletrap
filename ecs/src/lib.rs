use core::slice;
use hashbrown::HashMap;
use std::{
    any::{Any, TypeId},
    mem,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Entity(u64);

// Trait enabling dynamic dispatch for removal and safe downcasting
trait ComponentStorage: Any {
    fn remove_at(
        &mut self,
        index: usize,
        entity: Entity,
        entity_idxs: &mut HashMap<Entity, HashMap<TypeId, usize>>,
        type_id: TypeId,
    );
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct ComponentData<T> {
    data: Vec<(Entity, T)>,
}

impl<T: 'static> ComponentStorage for ComponentData<T> {
    fn remove_at(
        &mut self,
        index: usize,
        _entity: Entity,
        entity_idxs: &mut HashMap<Entity, HashMap<TypeId, usize>>,
        type_id: TypeId,
    ) {
        let vec = &mut self.data;
        let last_idx = vec.len() - 1;

        if index != last_idx {
            vec.swap(index, last_idx);
            let swapped_entity = vec[index].0;
            if let Some(swapped_idxs) = entity_idxs.get_mut(&swapped_entity) {
                swapped_idxs.insert(type_id, index);
            }
        }
        vec.pop();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct ComponentIterator<'a, T> {
    iter: slice::Iter<'a, (Entity, T)>,
}

impl<'a, T: 'static> Iterator for ComponentIterator<'a, T> {
    type Item = (Entity, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        // .next() on slice::IterMut gives us a &mut (Entity, T)
        // whose lifetime is correctly tied to the iterator borrow.
        self.iter
            .next()
            .map(|(entity, component)| (*entity, component))
    }
}

pub struct ComponentIteratorMut<'a, T> {
    iter: slice::IterMut<'a, (Entity, T)>,
}

impl<'a, T: 'static> Iterator for ComponentIteratorMut<'a, T> {
    type Item = (Entity, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(entity, component)| (*entity, component))
    }
}

pub struct EcsData {
    last_ent: u64,
    bank: HashMap<TypeId, Box<dyn ComponentStorage>>,
    entity_idxs: HashMap<Entity, HashMap<TypeId, usize>>,
}

impl EcsData {
    pub fn new() -> Self {
        Self {
            last_ent: 0,
            bank: Default::default(),
            entity_idxs: Default::default(),
        }
    }
    // Helpers taking only `bank` to allow disjoint field borrowing with `entity_idxs`
    fn get_comp_data_mut<T: 'static>(
        bank: &mut HashMap<TypeId, Box<dyn ComponentStorage>>,
    ) -> Option<&mut ComponentData<T>> {
        let storage = bank.get_mut(&TypeId::of::<T>())?;
        storage.as_any_mut().downcast_mut::<ComponentData<T>>()
    }

    fn get_comp_data<T: 'static>(
        bank: &HashMap<TypeId, Box<dyn ComponentStorage>>,
    ) -> Option<&ComponentData<T>> {
        let storage = bank.get(&TypeId::of::<T>())?;
        storage.as_any().downcast_ref::<ComponentData<T>>()
    }

    pub fn new_entity(&mut self) -> Entity {
        self.last_ent += 1;
        let new_ent = Entity(self.last_ent);
        self.entity_idxs.insert(new_ent, HashMap::new());
        new_ent
    }

    pub fn remove_entity(&mut self, entity: Entity) {
        let Some(idxs) = self.entity_idxs.remove(&entity) else {
            return;
        };

        for (type_id, idx) in idxs {
            if let Some(storage) = self.bank.get_mut(&type_id) {
                storage.remove_at(idx, entity, &mut self.entity_idxs, type_id);
            }
        }
    }

    pub fn insert_component<T: 'static>(&mut self, entity: Entity, mut elem: T) -> Option<T> {
        let type_id = TypeId::of::<T>();
        if !self.bank.contains_key(&type_id) {
            self.bank
                .insert(type_id, Box::new(ComponentData::<T> { data: vec![] }));
        }

        let old_idx = self
            .entity_idxs
            .get(&entity)
            .and_then(|h| h.get(&type_id).cloned());

        match old_idx {
            Some(old_idx) => {
                let data_obj = Self::get_comp_data_mut::<T>(&mut self.bank).unwrap();
                mem::swap(&mut data_obj.data[old_idx].1, &mut elem);
                Some(elem)
            }
            None => {
                let data_obj = Self::get_comp_data_mut::<T>(&mut self.bank).unwrap();
                let new_idx = data_obj.data.len();
                data_obj.data.push((entity, elem));
                self.entity_idxs
                    .entry(entity)
                    .or_default()
                    .insert(type_id, new_idx);
                None
            }
        }
    }

    pub fn get_component<T: 'static>(&self, entity: Entity) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        let old_idx = *self.entity_idxs.get(&entity)?.get(&type_id)?;
        let data = &Self::get_comp_data::<T>(&self.bank)?.data.get(old_idx)?.1;
        Some(data)
    }

    pub fn get_component_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        let old_idx = *self.entity_idxs.get(&entity)?.get(&type_id)?;
        let data = &mut Self::get_comp_data_mut::<T>(&mut self.bank)?
            .data
            .get_mut(old_idx)?
            .1;
        Some(data)
    }

    pub fn remove_component<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let type_id = TypeId::of::<T>();
        let old_idx = self.entity_idxs.get_mut(&entity)?.remove(&type_id)?;

        let comp_data = Self::get_comp_data_mut::<T>(&mut self.bank)?;

        let vec_last_idx = comp_data.data.len() - 1;
        if old_idx != vec_last_idx {
            comp_data.data.swap(old_idx, vec_last_idx);
            let swapped_entity = comp_data.data[old_idx].0;
            if let Some(swapped_idxs) = self.entity_idxs.get_mut(&swapped_entity) {
                swapped_idxs.insert(type_id, old_idx);
            }
        }
        let (_, elem) = comp_data.data.pop()?;
        Some(elem)
    }

    pub fn comp_data_vec<T: 'static>(&self) -> Option<ComponentIterator<'_, T>> {
        let comp_data = Self::get_comp_data::<T>(&self.bank)?;
        Some(ComponentIterator {
            iter: comp_data.data.iter(),
        })
    }

    pub fn comp_data_vec_mut<T: 'static>(&mut self) -> Option<ComponentIteratorMut<'_, T>> {
        let comp_data = Self::get_comp_data_mut::<T>(&mut self.bank)?;
        Some(ComponentIteratorMut {
            iter: comp_data.data.iter_mut(),
        })
    }
}
