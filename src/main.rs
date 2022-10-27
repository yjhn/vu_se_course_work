mod bench;
mod matrix;
mod tour;
mod tsp_problem;

use std::{env, path::Path};

use matrix::SquareMatrix;
use mpi::{
    collective::UserOperation,
    topology::SystemCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence, Root},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tour::{Length, Tour, TourIndex};
use tsp_problem::TspProblem;

const TEST_FILE: &str = "test_data/a10.tsp";
const EVOLUTION_GENERATION_COUNT: u32 = 10;
const POPULATION_COUNT: u32 = 8;
const INCREMENT: f64 = 1_f64 / POPULATION_COUNT as f64;
const EXCHANGE_GENERATIONS: u32 = 4;
const SOLUTION_FILE_NAME: &str = "solution.tsps";
// Maximum difference between two tour lengths to be considered 0.
// const TOUR_LENGTH_EPSILON: f64 = 0.001;
type RNG = SmallRng;

const GLOBAL_SEED: u64 = 865376825679;
const USE_HARDCODED_SEED: bool = false;

// Build and run locally:
// cargo build --release && RUST_BACKTRACE=1  mpirun --mca opal_warn_on_missing_libcuda 0 target/release/salesman test_data/a280.tsp

// TODO: set up ssh keys to directly connect to hpc
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    let is_root = rank == root_rank;
    if is_root {
        println!("World size: {size}");
    }

    // Broadcast global random seed.
    let mut global_seed_buf = if is_root {
        if USE_HARDCODED_SEED {
            [GLOBAL_SEED]
        } else {
            [rand::random()]
        }
    } else {
        [0]
    };
    root_process.broadcast_into(&mut global_seed_buf);
    let random_seed = global_seed_buf[0] + rank as u64;
    if is_root {
        println!("Global random seed: {random_seed}");
    }

    let path = get_input_file_path().unwrap_or_else(|| TEST_FILE.to_owned());
    if is_root {
        println!("File path: {path}");
    }

    let mut solver = TspSolver::<RNG>::from_file(path.clone(), random_seed, world);
    println!("Initial tour length: {}", solver.best_tour.length());

    solver.evolve(EVOLUTION_GENERATION_COUNT);

    println!(
        "Final tour length (for process at rank {rank}): {}",
        solver.best_tour.length()
    );

    // This must be executed by all processes.
    let best_global = solver.best_global_tour();

    // Root process outputs the results.
    if is_root {
        println!("Best global tour length: {}", best_global.length());
        best_global.save_to_file(path, SOLUTION_FILE_NAME);
        println!("Best tour saved to file {SOLUTION_FILE_NAME}");
    }
}

fn get_input_file_path() -> Option<String> {
    let mut args = env::args();
    // First arg is usually program path or empty.
    args.next();
    args.next()
}

// Position of city in all cities. Zero-based.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Equivalence)]
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
    current_generation: u32,
    rng: R,
    mpi: SystemCommunicator,
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(
        path: impl AsRef<Path>,
        random_seed: u64,
        mpi: SystemCommunicator,
    ) -> TspSolver<R> {
        let problem = TspProblem::from_file(path);
        Self::from_tsp_problem(problem, random_seed, mpi)
    }

    pub fn from_tsp_problem(
        problem: TspProblem,
        random_seed: u64,
        mpi: SystemCommunicator,
    ) -> TspSolver<R> {
        let mut probability_matrix = SquareMatrix::new(problem.number_of_cities(), 0.0);
        let mut rng = R::seed_from_u64(random_seed);
        let mut best_tour = Tour::PLACEHOLDER;

        // Generate POPULATION_COUNT random tours, optimize them and
        // update the prob matrix accordingly.
        for _ in 0..POPULATION_COUNT {
            let mut opt_tour =
                Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);

            opt_tour.two_opt_take_best_each_time(problem.distances());

            Self::update_probabilitities::<true>(&mut probability_matrix, &opt_tour);

            if opt_tour.is_shorter_than(&best_tour) {
                best_tour = opt_tour;
            }
        }

        TspSolver {
            problem,
            probability_matrix,
            best_tour,
            current_generation: 0,
            rng,
            mpi,
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
        for gen in 0..generations {
            self.current_generation += 1;

            self.evolve_inner();
            if gen % EXCHANGE_GENERATIONS == 0 {
                self.exchange_best_tours();
            }
        }
    }

    fn exchange_best_tours(&mut self) {
        let global_best = self.best_global_tour();

        // Update the probability matrix if the global best tour is
        // shorter than local best tour (one process will have its tour
        // chosen as the global best tour).
        if global_best.is_shorter_than(&self.best_tour) {
            Self::update_probabilitities::<false>(&mut self.probability_matrix, &self.best_tour);
            Self::update_probabilitities::<true>(&mut self.probability_matrix, &global_best);
        }

        // TODO: maybe it's worth it to also update our local best tour?
        // The paper doesn't do it for some reason.
    }

    fn best_global_tour(&mut self) -> Tour {
        // Exchange best tours.
        // for now don't exchange tour length
        // let rank = self.mpi.rank();
        let mpi_closure = UserOperation::commutative(|read_buf, write_buf| {
            let local_best = read_buf.downcast::<CityIndex>().unwrap();
            let global_best = write_buf.downcast::<CityIndex>().unwrap();

            let local_tour_length = local_best.hack_get_tour_length_from_last_element();

            let global_tour_length = global_best.hack_get_tour_length_from_last_element();

            // println!("Exchanging global tours, at rank {rank}. GBTL: {global_tour_length}, LBTL: {local_tour_length}");

            if local_tour_length < global_tour_length {
                global_best.copy_from_slice(local_best);
            }
        });

        // Append tour length at the end of the tour to not calculate
        // it needlessly.
        self.best_tour.hack_append_length_at_tour_end();

        // This needs to be filled up before exchanging.
        let mut best_global_tour: Vec<CityIndex> = self.best_tour.cities().to_owned();

        self.mpi
            .all_reduce_into(self.best_tour.cities(), &mut best_global_tour, &mpi_closure);

        // Remove tour length hack.
        self.best_tour.remove_hack_length();

        Tour::from_hack_cities(best_global_tour)
    }

    fn evolve_inner(&mut self) {
        let loser = Tour::gen_tour_from_prob_matrix(
            self.number_of_cities(),
            &self.probability_matrix,
            self.problem.distances(),
            &mut self.rng,
        );
        let mut winner = loser.clone();

        winner.two_opt_take_best_each_time(self.distances());

        // Increase probs of all paths taken by the winner and
        // decrease probs of all paths taken by the loser.
        Self::update_probabilitities::<true>(&mut self.probability_matrix, &winner);
        Self::update_probabilitities::<false>(&mut self.probability_matrix, &loser);
        if winner.is_shorter_than(&self.best_tour) {
            self.best_tour = winner;
            println!(
                "New best tour length in generation {}: {}",
                self.current_generation,
                self.best_tour.length()
            );
        }
    }

    fn distance(&self, a: usize, b: usize) -> f64 {
        self.problem.distances()[(a, b)]
    }

    fn update_probabilitities<const INC: bool>(prob_matrix: &mut SquareMatrix<f64>, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                let (l, h) = order(c1.get(), c2.get());
                if INC {
                    prob_matrix[(h, l)] += INCREMENT;
                } else {
                    prob_matrix[(h, l)] -= INCREMENT;
                }
                // All values in probability matrix must always be in range [0..1].
                prob_matrix[(h, l)] = f64::clamp(prob_matrix[(h, l)], 0.0, 1.0)
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.get_path(TourIndex::new(0), TourIndex::new(t.city_count() - 1));
        let (l, h) = order(c1.get(), c2.get());

        if INC {
            prob_matrix[(h, l)] += INCREMENT;
        } else {
            prob_matrix[(h, l)] -= INCREMENT;
        }
        // All values in probability matrix must always be in range [0..1].
        prob_matrix[(h, l)] = f64::clamp(prob_matrix[(h, l)], 0.0, 1.0)
    }
}

// Returns (low, high).
pub fn order(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}
