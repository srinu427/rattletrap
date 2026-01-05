use hashbrown::HashMap;
use winit::{event::ElementState, keyboard::PhysicalKey};

pub struct KeyState {
    last_pressed: Option<std::time::Instant>,
    last_released: Option<std::time::Instant>,
    is_pressed: bool,
    pressed_this_frame: bool,
    released_this_frame: bool,
}

impl KeyState {
    pub fn new(state: ElementState) -> Self {
        if state.is_pressed() {
            Self {
                is_pressed: true,
                last_released: None,
                last_pressed: Some(std::time::Instant::now()),
                pressed_this_frame: true,
                released_this_frame: false,
            }
        } else {
            Self {
                is_pressed: false,
                last_released: Some(std::time::Instant::now()),
                last_pressed: None,
                pressed_this_frame: false,
                released_this_frame: false,
            }
        }
    }

    pub fn update(&mut self, state: ElementState) {
        if state.is_pressed() {
            if !self.is_pressed {
                self.is_pressed = true;
                self.pressed_this_frame = true;
                self.last_pressed = Some(std::time::Instant::now());
            }
        } else {
            if self.is_pressed {
                self.is_pressed = false;
                self.released_this_frame = true;
                self.last_released = Some(std::time::Instant::now());
            }
        }
    }
}

pub struct Inputs {
    keys: HashMap<PhysicalKey, KeyState>,
    mouse: (f64, f64),
}

impl Inputs {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            mouse: (0.0, 0.0),
        }
    }

    pub fn advance_frame(&mut self) {
        for ks in self.keys.values_mut() {
            ks.pressed_this_frame = false;
            ks.released_this_frame = false;
        }
        self.mouse = (0.0, 0.0);
    }

    pub fn add_key_event(&mut self, key: PhysicalKey, state: ElementState) {
        match self.keys.get_mut(&key) {
            Some(ke) => {
                ke.update(state);
            }
            None => {
                self.keys.insert(key, KeyState::new(state));
            }
        };
    }

    pub fn key_pressed(&self, key: PhysicalKey) -> bool {
        match self.keys.get(&key) {
            Some(s) => s.is_pressed,
            None => false,
        }
    }

    pub fn key_pressed_this_frame(&self, key: PhysicalKey) -> bool {
        match self.keys.get(&key) {
            Some(s) => s.is_pressed && s.pressed_this_frame,
            None => false,
        }
    }

    pub fn add_mouse_delta(&mut self, delta: (f64, f64)) {
        self.mouse.0 += delta.0;
        self.mouse.1 += delta.1;
    }

    pub fn mouse_delta(&self) -> (f64, f64) {
        self.mouse
    }

    pub fn reset_mouse(&mut self) {
        self.mouse.0 = 0.0;
        self.mouse.1 = 0.0;
    }
}
