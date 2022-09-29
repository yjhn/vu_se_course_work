mod bench;
mod matrix;
mod tour;
mod tsp_problem;

use std::path::Path;

use matrix::SquareMatrix;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tour::Tour;
use tsp_problem::TspProblem;

const TEST_FILE: &str = "test_data/a10.tsp";
const RANDOM_SEED: u64 = 1543434354;
const EVOLUTION_GENERATION_COUNT: u32 = 10;
const POPULATION_COUNT: u32 = 100;

fn main() {
    let mut solver = TspSolver::<SmallRng>::from_file(TEST_FILE, RANDOM_SEED);
    solver.random_as_best();
}

// Position of city in all cities. Zero-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CityIndex(usize);

impl CityIndex {
    pub fn new(index: usize) -> CityIndex {
        CityIndex(index)
    }

    pub fn get(&self) -> usize {
        self.0
    }
}

pub struct TspSolver<R: Rng + SeedableRng> {
    problem: TspProblem,
    probability_matrix: SquareMatrix<f64>,
    best_tour: Tour,
    rng: R,
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(path: impl AsRef<Path>, random_seed: u64) -> TspSolver<R> {
        let problem = TspProblem::from_file(path);
        let probability_matrix = SquareMatrix::new(problem.number_of_cities(), 0.5);

        TspSolver {
            problem,
            probability_matrix,
            best_tour: Tour::PLACEHOLDER,
            rng: R::seed_from_u64(random_seed),
        }
    }

    pub fn from_tsp_problem(problem: TspProblem, random_seed: u64) -> TspSolver<R> {
        let probability_matrix = SquareMatrix::new(problem.number_of_cities(), 0.5);
        TspSolver {
            problem,
            probability_matrix,
            best_tour: Tour::PLACEHOLDER,
            rng: R::seed_from_u64(random_seed),
        }
    }

    pub fn random_as_best(&mut self) {
        self.best_tour = self.random_tour();
    }

    pub fn number_of_cities(&self) -> usize {
        self.problem.number_of_cities()
    }

    pub fn distances(&self) -> &SquareMatrix<f64> {
        self.problem.distances()
    }

    // This is O(n^2), but I don't know how to optimize it.
    pub fn random_tour(&mut self) -> Tour {
        let len = self.number_of_cities();

        let mut cities = Vec::with_capacity(len);
        let start = self.rng.gen_range(0..len);
        cities.push(CityIndex::new(start));
        for idx in 1..len {
            // Generate indices in unused cities only to avoid duplicates.
            let index = self.rng.gen_range(0..(len - idx));
            // Check for unused cities and choose index-th unused city.
            let mut unused_city_count = 0;
            for c in 0..len {
                let city_index = CityIndex(c);
                if !cities.contains(&city_index) {
                    if unused_city_count == index {
                        cities.push(city_index);
                        break;
                    }
                    unused_city_count += 1;
                }
            }
        }

        Tour::from_cities(cities, &self.distances())
    }

    pub fn optimize_tour(&self, tour: &mut Tour) {
        //TODO: add LK limited to 2-opt
    }

    pub fn evolve(&mut self, generations: u32) {
        //TODO: evolution
    }
}
