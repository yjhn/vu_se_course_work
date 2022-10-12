mod bench;
mod matrix;
mod tour;
mod tsp_problem;

use std::{cmp::Ordering, path::Path};

use matrix::SquareMatrix;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tour::{Tour, TourIndex};
use tsp_problem::TspProblem;

const TEST_FILE: &str = "test_data/a10.tsp";
const RANDOM_SEED: u64 = 1543434354;
const EVOLUTION_GENERATION_COUNT: u32 = 10;
const POPULATION_COUNT: u32 = 100;
const INCREMENT: f64 = 1 as f64 / POPULATION_COUNT as f64;

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
    // Only upper left triangle will be used.
    probability_matrix: SquareMatrix<f64>,
    best_tour: Tour,
    current_tour: Tour,
    current_generation: u32,
    rng: R,
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(path: impl AsRef<Path>, random_seed: u64) -> TspSolver<R> {
        let problem = TspProblem::from_file(path);
        Self::from_tsp_problem(problem, random_seed)
    }

    pub fn from_tsp_problem(problem: TspProblem, random_seed: u64) -> TspSolver<R> {
        let mut probability_matrix = SquareMatrix::new(problem.number_of_cities(), 0.0);
        let mut rng = R::seed_from_u64(random_seed);
        let current_tour = Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);

        // Generate POPULATION_COUNT random cities, optimize them and
        // update the prob matrix accordingly.
        for _ in 0..POPULATION_COUNT {
            let mut opt_tour =
                Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);
            opt_tour.ls_2_opt_take_best(problem.distances());

            Self::update_probabilitities::<true>(&mut probability_matrix, &opt_tour);
        }

        TspSolver {
            problem,
            probability_matrix,
            best_tour: Tour::PLACEHOLDER,
            current_tour,
            current_generation: 0,
            rng,
        }
    }

    pub fn random_as_best(&mut self) {
        self.best_tour = Tour::random(
            self.number_of_cities(),
            self.problem.distances(),
            &mut self.rng,
        );
    }

    pub fn number_of_cities(&self) -> usize {
        self.problem.number_of_cities()
    }

    pub fn distances(&self) -> &SquareMatrix<f64> {
        self.problem.distances()
    }

    pub fn evolve(&mut self, generations: u32) {
        //TODO: evolution
        for _ in 0..generations {
            self.current_generation += 1;
        }
    }

    fn gen_tour_from_prob_matrix(&mut self) -> Tour {
        todo!()
    }

    // Kinda useless, since this is not used in the paper it seems.
    // pub fn update_probabilities(&mut self, t1: &Tour, t2: &Tour) {
    //     // Update each path to have higher probability if
    //     // the winner has it and lower otherwise.
    //     if t1.is_shorter_than(t2) {
    //         self.update_probabilitities::<true>(t1);
    //         self.update_probabilitities::<false>(t2);
    //     } else {
    //         self.update_probabilitities::<true>(t2);
    //         self.update_probabilitities::<false>(t1);
    //     }
    // }

    fn update_probabilitities<const INC: bool>(prob_matrix: &mut SquareMatrix<f64>, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                let (l, h) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                if INC {
                    prob_matrix[(h.get(), l.get())] += INCREMENT;
                } else {
                    prob_matrix[(h.get(), l.get())] -= INCREMENT;
                }
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.get_path(TourIndex::new(0), TourIndex::new(t.city_count() - 1));
        let (l, h) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        if INC {
            prob_matrix[(h.get(), l.get())] += INCREMENT;
        } else {
            prob_matrix[(h.get(), l.get())] -= INCREMENT;
        }
    }
}

pub struct TwoOptTspSolver {
    problem: TspProblem,
    tour: Tour,
}

impl TwoOptTspSolver {
    pub fn new(tsp_problem: TspProblem, random_seed: u64) -> TwoOptTspSolver {
        // Construct a tour using nearest-neighbour algorithm.
        todo!()
    }
}

trait TspSolve {
    fn new(tsp_problem: TspProblem, random_seed: u64) -> Self;

    fn solve(&mut self) -> Tour;
}
