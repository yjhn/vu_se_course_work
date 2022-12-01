// Allow dead code for now.
#![allow(dead_code)]

mod bench;
mod benchmark;
mod matrix;
mod probability_matrix;
mod tour;
mod tsp_problem;
mod tsp_solver;

use std::{env, fmt::Display, path::Path};

use mpi::{
    topology::SystemCommunicator,
    traits::{Communicator, Root},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::tsp_solver::TspSolver;

mod config {
    use crate::SolutionStrategy;
    use rand::rngs::SmallRng;

    pub const TEST_FILE: &str = "test_data/a10.tsp";
    pub const SOLUTION_FILE_NAME: &str = "solution.tsps";
    pub type MainRng = SmallRng;
    pub const GLOBAL_SEED: u64 = 865376825679;
    pub const USE_HARDCODED_SEED: bool = false;

    // Algorithm constants
    pub const EVOLUTION_GENERATION_COUNT: u32 = 50;
    pub const POPULATION_COUNT: u32 = 32;
    pub const INCREMENT: f64 = 1_f64 / POPULATION_COUNT as f64;
    pub const EXCHANGE_GENERATIONS: u32 = 4;
    // const SOLUTION_STRATEGY: SolutionStrategy = SolutionStrategy::Cga;
    pub const SOLUTION_STRATEGY: SolutionStrategy = SolutionStrategy::CgaTwoOpt;
    pub const EPSILON: f64 = 0.0000001;

    pub const BENCHMARK: bool = false;
    // Benchmarking constants
    pub mod benchmark {
        pub const MIN_PROCESSES: u32 = 1;
        pub const MAX_PROCESSES: u32 = 10;
        pub const MIN_GENERATIONS: u32 = 1;
        pub const MAX_GENERATIONS: u32 = 200;
        pub const REPEAT_TIMES: u32 = 10;
    }
}

// Build and run locally on a single thread:
// cargo build --release && RUST_BACKTRACE=1  mpirun -c 1 --use-hwthread-cpus --mca opal_warn_on_missing_libcuda 0 target/release/salesman test_data/a280.tsp

fn main() {
    // Initialize stuff.
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

    if config::BENCHMARK {
        use config::benchmark::*;

        benchmark::benchmark::<&str, SmallRng>(
            &path,
            random_seed,
            world,
            rank,
            is_root,
            MIN_PROCESSES,
            MAX_PROCESSES,
            MIN_GENERATIONS,
            MAX_GENERATIONS,
            REPEAT_TIMES,
        )
    } else {
        run::<&str, SmallRng>(
            &path,
            config::SOLUTION_STRATEGY,
            config::EVOLUTION_GENERATION_COUNT,
            random_seed,
            world,
            rank,
            is_root,
            config::SOLUTION_FILE_NAME,
        );
    }
}

fn get_input_file_path() -> Option<String> {
    let mut args = env::args();
    // First arg is usually program path or empty.
    args.next();
    args.next()
}

fn run<PD, R>(
    path: PD,
    solution_strategy: SolutionStrategy,
    evolution_generation_count: u32,
    random_seed: u64,
    world: SystemCommunicator,
    rank: i32,
    is_root: bool,
    solution_file_name: &str,
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    let mut solver = TspSolver::<R>::from_file(&path, solution_strategy, random_seed, world);
    println!("Initial tour length: {}", solver.best_tour_length());

    solver.evolve(evolution_generation_count);

    println!(
        "Final tour length (for process at rank {rank}): {}",
        solver.best_tour_length()
    );

    // This must be executed by all processes.
    let best_global = solver.best_global_tour();

    // Root process outputs the results.
    if is_root {
        println!("Best global tour length: {}", best_global.length());
        best_global.save_to_file(path, solution_file_name);
        println!("Best tour saved to file {}", solution_file_name);
    }
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
