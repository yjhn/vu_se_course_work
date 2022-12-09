use std::time::Duration;

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
}
