// Allow dead code for now.
#![allow(dead_code)]

mod bench;
mod matrix;
mod probability_matrix;
mod tour;
mod tsp_problem;
mod tsp_solver;

use std::env;

use mpi::traits::{Communicator, Root};

use crate::tsp_solver::TspSolver;

mod config {
    use rand::rngs::SmallRng;

    use crate::SolutionStrategy;

    pub const TEST_FILE: &str = "test_data/a10.tsp";
    pub const EVOLUTION_GENERATION_COUNT: u32 = 50;
    pub const POPULATION_COUNT: u32 = 128;
    pub const INCREMENT: f64 = 1_f64 / POPULATION_COUNT as f64;
    pub const EXCHANGE_GENERATIONS: u32 = 4;
    // const SOLUTION_STRATEGY: SolutionStrategy = SolutionStrategy::Cga;
    pub const SOLUTION_STRATEGY: SolutionStrategy = SolutionStrategy::CgaTwoOpt;
    pub const SOLUTION_FILE_NAME: &str = "solution.tsps";
    // Maximum difference between two tour lengths to be considered 0.
    // const TOUR_LENGTH_EPSILON: f64 = 0.001;

    pub const GLOBAL_SEED: u64 = 865376825679;
    pub const USE_HARDCODED_SEED: bool = false;

    pub type MainRng = SmallRng;
}

// Build and run locally:
// cargo build --release && RUST_BACKTRACE=1  mpirun --mca opal_warn_on_missing_libcuda 0 target/release/salesman test_data/a280.tsp

fn main() {
    const MPI_ROOT_RANK: i32 = 0;
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_process = world.process_at_rank(MPI_ROOT_RANK);
    let is_root = rank == MPI_ROOT_RANK;
    if is_root {
        println!("World size: {size}");
    }

    // Broadcast global random seed.
    let mut global_seed_buf = if is_root {
        if config::USE_HARDCODED_SEED {
            [config::GLOBAL_SEED]
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

    let path = get_input_file_path().unwrap_or_else(|| config::TEST_FILE.to_owned());
    if is_root {
        println!("File path: {path}");
    }

    let mut solver = TspSolver::<config::MainRng>::from_file(
        path.clone(),
        config::SOLUTION_STRATEGY,
        random_seed,
        world,
    );
    println!("Initial tour length: {}", solver.best_tour_length());

    solver.evolve(config::EVOLUTION_GENERATION_COUNT);

    println!(
        "Final tour length (for process at rank {rank}): {}",
        solver.best_tour_length()
    );

    // This must be executed by all processes.
    let best_global = solver.best_global_tour();

    // Root process outputs the results.
    if is_root {
        println!("Best global tour length: {}", best_global.length());
        best_global.save_to_file(path, config::SOLUTION_FILE_NAME);
        println!("Best tour saved to file {}", config::SOLUTION_FILE_NAME);
    }
}

fn get_input_file_path() -> Option<String> {
    let mut args = env::args();
    // First arg is usually program path or empty.
    args.next();
    args.next()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionStrategy {
    Cga,
    CgaTwoOpt,
    CgaThreeOpt,
}

// Returns (low, high).
pub fn order(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}
