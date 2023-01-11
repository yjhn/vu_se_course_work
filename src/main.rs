// Allow dead code for now to not get overwhelmed by warnings.
#![allow(dead_code)]

mod arguments;
mod benchmark;
mod matrix;
mod probability_matrix;
mod timing;
mod tour;
mod tsp_problem;
mod tsp_solver;

use std::{fmt::Display, path::Path};

use clap::Parser;
use mpi::{
    topology::{Process, SystemCommunicator},
    traits::{Communicator, Root},
};
use rand::{Rng, SeedableRng};

use crate::{arguments::Args, tsp_solver::TspSolver};

mod config {
    use rand::rngs::SmallRng;

    pub const SOLUTION_FILE_NAME: &str = "solution.tsps";
    pub type MainRng = SmallRng;

    // Algorithm constants
    pub const EVOLUTION_GENERATION_COUNT: u32 = 50;
    pub const POPULATION_SIZE: u32 = 32;
    pub const INCREMENT: f64 = 1_f64 / POPULATION_SIZE as f64;
    pub const EXCHANGE_GENERATIONS: u32 = 4;

    pub const BENCHMARK: bool = true;
    // Benchmarking constants
    pub mod benchmark {
        pub const MAX_GENERATIONS: u32 = 500;
        // As defined in the paper.
        pub const REPEAT_TIMES: u32 = 10;
        pub const POPULATION_SIZE: u32 = 128;
        pub const EXCHANGE_GENERATIONS: [u32; 4] = [4, 8, 16, 32];
        pub const RESULTS_DIR: &str = "benchmark_results";
    }

    pub mod solution_length {
        pub const ATT532: u32 = 27686;
        pub const GR666: u32 = 294358;
        pub const RAT783: u32 = 8806;
        pub const PR1002: u32 = 259045;
    }
}

// Build and run locally on a single thread:
// RUSTFLAGS="-C target-cpu=native" cargo build --release && RUST_BACKTRACE=1 mpirun -c 1 --use-hwthread-cpus --mca opal_warn_on_missing_libcuda 0 target/release/salesman -f data/att532.tsp -b 5 -a CgaTwoOpt --benchmark

fn main() {
    // Initialize stuff.
    const MPI_ROOT_RANK: i32 = 0;
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_process = world.process_at_rank(MPI_ROOT_RANK);
    let is_root = rank == MPI_ROOT_RANK;

    let Args {
        files,
        max_generations,
        benchmark,
        bench_repeat_times,
        algorithms,
        bench_results_dir,
        population_size,
        exchange_generations,
    } = arguments::Args::parse();

    let algorithms: Vec<Algorithm> = algorithms
        .iter()
        .map(|a| a.as_str().try_into().unwrap())
        .collect();

    if is_root {
        println!("TSP files:\n{files:?}");
        println!("Max generations: {max_generations}");
        println!("Benchmark: {benchmark}");
        println!("Benchmark repeat times: {bench_repeat_times}");
        println!("Algorithms: {algorithms:?}");
        println!("Benchmark results directory: {bench_results_dir}");
        println!("Population size: {population_size}");
        println!("Exchange generations: {exchange_generations:?}");
    }

    if benchmark {
        use config::benchmark::*;
        use config::solution_length;

        for path in files {
            let (problem_name, solution_length) = match path.split('/').last().unwrap() {
                "att532.tsp" => ("att532", solution_length::ATT532),
                "gr666.tsp" => ("gr666", solution_length::GR666),
                "rat783.tsp" => ("rat783", solution_length::RAT783),
                "pr1002.tsp" => ("pr1002", solution_length::PR1002),
                _ => panic!("Benchmark only works with att532, gr666, rat783 and pr1002"),
            };

            let exc_gen = if exchange_generations.is_empty() {
                &EXCHANGE_GENERATIONS
            } else {
                exchange_generations.as_slice()
            };

            benchmark::benchmark::<&str, config::MainRng>(
                &path,
                problem_name,
                solution_length,
                max_generations,
                world,
                root_process,
                rank,
                is_root,
                bench_repeat_times,
                population_size,
                exc_gen,
                &algorithms,
                &bench_results_dir,
            );
        }
    } else {
        if is_root {
            println!("World size: {size}");
        }

        for path in files {
            if is_root {
                println!("File path: {path}");
            }
            // Separate random seed must be created for each run.
            let random_seed = initialize_random_seed(root_process, rank, is_root);

            run::<&str, config::MainRng>(
                &path,
                algorithms[0],
                config::EVOLUTION_GENERATION_COUNT,
                config::EXCHANGE_GENERATIONS,
                config::POPULATION_SIZE,
                random_seed,
                world,
                rank,
                is_root,
                config::SOLUTION_FILE_NAME,
            );
        }
    }
}

fn initialize_random_seed(
    root_process: Process<SystemCommunicator>,
    rank: i32,
    is_root: bool,
) -> u64 {
    // Broadcast global random seed.
    let mut global_seed_buf = if is_root { [rand::random()] } else { [0] };
    root_process.broadcast_into(&mut global_seed_buf);
    let random_seed = global_seed_buf[0] + rank as u64;
    // if is_root {
    // println!("Global random seed: {random_seed}");
    // }
    random_seed
}

fn run<PD, R>(
    path: PD,
    solution_strategy: Algorithm,
    evolution_generation_count: u32,
    exchange_generations: u32,
    population_size: u32,
    random_seed: u64,
    world: SystemCommunicator,
    rank: i32,
    is_root: bool,
    solution_file_name: &str,
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    let mut solver = TspSolver::<R>::from_file(
        &path,
        solution_strategy,
        random_seed,
        world,
        exchange_generations,
        population_size,
    );
    println!("Initial tour length: {}", solver.best_tour_length());

    solver.evolve(evolution_generation_count);

    println!(
        "Final tour length (for process at rank {rank}): {}",
        solver.best_tour_length()
    );

    // This must be executed by all processes.
    let (best_global, _avg_exchange_duration) = solver.best_global_tour();

    // Root process outputs the results.
    if is_root {
        println!("Best global tour length: {}", best_global.length());
        best_global.save_to_file(path, solution_file_name);
        println!("Best tour saved to file {}", solution_file_name);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Cga,
    CgaTwoOpt,
    CgaThreeOpt,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::Cga => f.write_str("cga"),
            Algorithm::CgaTwoOpt => f.write_str("cga2opt"),
            Algorithm::CgaThreeOpt => f.write_str("cga3opt"),
        }
    }
}

impl TryFrom<&str> for Algorithm {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "Cga" => Ok(Algorithm::Cga),
            "CgaTwoOpt" => Ok(Algorithm::CgaTwoOpt),
            "CgaThreeOpt" => Ok(Algorithm::CgaThreeOpt),
            _ => Err(()),
        }
    }
}

// Returns (low, high).
pub fn order<T: Ord>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}
