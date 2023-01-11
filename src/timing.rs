use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct SingleGenerationTimingInfo {
    tour_generation_from_prob_matrix: Duration,
    tour_optimization: Duration,
}

impl SingleGenerationTimingInfo {
    pub fn new(gen: Duration, opt: Duration) -> SingleGenerationTimingInfo {
        SingleGenerationTimingInfo {
            tour_generation_from_prob_matrix: gen,
            tour_optimization: opt,
        }
    }

    pub fn tour_generation_from_prob_matrix(&self) -> Duration {
        self.tour_generation_from_prob_matrix
    }

    pub fn tour_optimization(&self) -> Duration {
        self.tour_optimization
    }
}

pub struct GenerationsInfo {
    tour_generation_from_prob_matrix: Vec<Duration>,
    tour_optimization: Vec<Duration>,
    optimized_length: Vec<u32>,
}

impl GenerationsInfo {
    pub fn new(capacity: usize) -> GenerationsInfo {
        GenerationsInfo {
            tour_generation_from_prob_matrix: Vec::with_capacity(capacity),
            tour_optimization: Vec::with_capacity(capacity),
            optimized_length: Vec::with_capacity(capacity),
        }
    }

    pub fn add_generation(
        &mut self,
        gen_from_prob_matrix: Duration,
        optimization: Duration,
        optimized_length: u32,
    ) {
        self.tour_generation_from_prob_matrix
            .push(gen_from_prob_matrix);
        self.tour_optimization.push(optimization);
        self.optimized_length.push(optimized_length);
    }

    pub fn tour_generation_from_prob_matrix(&self) -> &[Duration] {
        &self.tour_generation_from_prob_matrix
    }

    pub fn tour_optimization(&self) -> &[Duration] {
        self.tour_optimization.as_ref()
    }

    pub fn optimized_length(&self) -> &[u32] {
        self.optimized_length.as_ref()
    }

    pub fn len(&self) -> usize {
        self.tour_generation_from_prob_matrix.len()
    }

    /// Extends the Vecs to have exactly `total_len` records
    /// by duplicating the last record.
    pub fn extend_duplicate(&mut self, total_len: usize) {
        let last_1 = *self.tour_generation_from_prob_matrix.last().unwrap();
        let last_2 = *self.tour_optimization.last().unwrap();
        let last_3 = *self.optimized_length.last().unwrap();

        for _ in 0..(total_len - self.len()) {
            self.add_generation(last_1, last_2, last_3);
        }
    }
}
