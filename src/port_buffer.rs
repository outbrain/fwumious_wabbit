#[derive(Clone, Debug)]
pub struct PortBuffer {
    pub tape: Vec<f32>,
    pub observations: Vec<f32>,
    pub tape_len: usize,
}

impl PortBuffer {
    pub fn new(tape_len: usize) -> PortBuffer {
        PortBuffer {
            tape: Default::default(),
            observations: Default::default(),
            tape_len,
        }
    }

    pub fn reset(&mut self) {
        self.observations.truncate(0);
        self.tape.resize(self.tape_len, 0.0);
    }
}
