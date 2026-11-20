use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

type Origin = [u64; 3];

pub(crate) struct ChunkCache {
    capacity: usize,
    used: usize,
    clock: u128,
    order: BTreeMap<u128, Origin>,
    entries: HashMap<Origin, (u128, Arc<Vec<u8>>)>,
}

impl ChunkCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            clock: 0,
            order: BTreeMap::new(),
            entries: HashMap::new(),
        }
    }

    pub(crate) fn get(&mut self, origin: Origin) -> Option<Arc<Vec<u8>>> {
        let (age, data) = self.entries.get_mut(&origin)?;
        self.order.remove(age);
        self.clock += 1;
        *age = self.clock;
        self.order.insert(*age, origin);
        Some(Arc::clone(data))
    }

    pub(crate) fn insert(&mut self, origin: Origin, data: Vec<u8>) {
        if data.capacity() > self.capacity || self.get(origin).is_some() {
            return;
        }
        while data.capacity() > self.capacity - self.used {
            let (_, oldest) = self.order.pop_first().expect("used cache has entries");
            let (_, removed) = self.entries.remove(&oldest).expect("tracked cache entry");
            self.used -= removed.capacity();
        }
        self.clock += 1;
        self.used += data.capacity();
        self.order.insert(self.clock, origin);
        self.entries.insert(origin, (self.clock, Arc::new(data)));
    }
}
