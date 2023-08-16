#[derive(Clone, Debug)]
pub struct PortBuffer {
    pub tape: Vec<f32>,
    pub observations: Vec<f32>,
    pub tape_len: usize,
    pub monte_carlo_stats: Option<MonteCarloStats>,
}

#[derive(Clone, Debug)]
pub struct MonteCarloStats {
    pub mean: f32,
    pub variance: f32,
    pub standard_deviation: f32,
}

impl PortBuffer {
    pub fn new(tape_len: usize) -> PortBuffer {
        PortBuffer {
            tape: Default::default(),
            observations: Default::default(),
            tape_len,
            monte_carlo_stats: None,
        }
    }

    pub fn reset(&mut self) {
        self.observations.truncate(0);
        self.tape.resize(self.tape_len, 0.0);
        self.monte_carlo_stats = None;
    }
}
