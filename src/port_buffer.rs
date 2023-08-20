#[derive(Clone, Debug)]
pub struct PortBuffer {
    pub tape: Vec<f32>,
    pub observations: Vec<f32>,
    pub tape_len: usize,
    pub stats: Option<PredictionStats>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictionStats {
    pub mean: f32,
    pub variance: f32,
    pub standard_deviation: f32,
    pub count: usize,
}

impl PortBuffer {
    pub fn new(tape_len: usize) -> PortBuffer {
        PortBuffer {
            tape: Default::default(),
            observations: Default::default(),
            tape_len,
            stats: None,
        }
    }

    pub fn reset(&mut self) {
        self.observations.truncate(0);
        self.tape.resize(self.tape_len, 0.0);
        self.stats = None;
    }
}
