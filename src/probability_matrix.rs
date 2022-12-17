use std::ops::Index;

use crate::{config, matrix::SquareMatrix, order, tour::Tour, tsp_solver::CityIndex};

pub struct ProbabilityMatrix(SquareMatrix<f64>);

impl ProbabilityMatrix {
    pub fn new(side_length: usize, init_value: f64) -> ProbabilityMatrix {
        ProbabilityMatrix(SquareMatrix::new(side_length, init_value))
    }

    pub fn side_length(&self) -> usize {
        self.0.side_length()
    }

    pub fn decrease_probabilitities(&mut self, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                self.decrease_prob(c1, c2);
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.last_to_first_path();
        self.decrease_prob(c1, c2);
    }

    fn decrease_prob(&mut self, c1: CityIndex, c2: CityIndex) {
        let (l, h) = order(c1.get(), c2.get());
        let val = self.0[(h, l)] - config::INCREMENT;
        // All values in probability matrix must always be in range [0..1].
        self.0[(h, l)] = f64::clamp(val, 0.0, 1.0)
    }

    pub fn increase_probabilitities(&mut self, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                self.increase_prob(c1, c2);
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.last_to_first_path();
        self.increase_prob(c1, c2);
    }

    fn increase_prob(&mut self, c1: CityIndex, c2: CityIndex) {
        let (l, h) = order(c1.get(), c2.get());
        let val = self.0[(h, l)] + config::INCREMENT;
        // All values in probability matrix must always be in range [0..1].
        self.0[(h, l)] = f64::clamp(val, 0.0, 1.0)
    }
}

// First number must be higher than second.
impl Index<(usize, usize)> for ProbabilityMatrix {
    type Output = f64;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        debug_assert!(x > y);

        &self.0[(x, y)]
    }
}

// First number must be higher than second.
impl Index<(CityIndex, CityIndex)> for ProbabilityMatrix {
    type Output = f64;

    fn index(&self, (x, y): (CityIndex, CityIndex)) -> &Self::Output {
        debug_assert!(x > y);

        &self.0[(x.get(), y.get())]
    }
}
