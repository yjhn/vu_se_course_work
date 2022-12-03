use std::fs;
use std::io::Write;
use std::{fmt::Display, fs::File, path::Path};

use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::Communicator;
use rand::{Rng, SeedableRng};

use crate::initialize_random_seed;
use crate::{tsp_solver::TspSolver, SolutionStrategy};

pub fn benchmark<PD, R>(
    path: PD,
    problem_name: &str,
    solution_length: u32,
    max_generations: u32,
    world: SystemCommunicator,
    root_process: Process<SystemCommunicator>,
    rank: i32,
    is_root: bool,
    repeat_times: u32,
    population_size: u32,
    exchange_generations: [u32; 4],
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    // Create benchmark results directory.
    if !Path::new("benchmark_results").exists() {
        fs::create_dir("benchmark_results").unwrap();
    }

    // Write everything to a single file for simplicity.
    let save_file_name = format!(
        "benchmark_results/bm_{problem_name}_{}_cpus.out",
        world.size()
    );
    let mut file = File::create(save_file_name).unwrap();
    writeln!(file, "Problem file name: {path}").unwrap();
    writeln!(file, "Problem name: {problem_name}").unwrap();
    writeln!(file, "MPI processes: {}", world.size()).unwrap();

    if is_root {
        println!("Problem file name: {path}");
        println!("Problem name: {problem_name}");
        println!("MPI processes: {}", world.size());
    }

    for s in [
        SolutionStrategy::Cga,
        SolutionStrategy::CgaTwoOpt,
        SolutionStrategy::CgaThreeOpt,
    ] {
        writeln!(file, "Algorithm used: {s:?}").unwrap();

        for exc in exchange_generations {
            for i in 0..repeat_times {
                // Separate random seed must be created for each run.
                let random_seed = initialize_random_seed(root_process, rank, is_root);

                let mut solver =
                    TspSolver::<R>::from_file(&path, s, random_seed, world, exc, population_size);

                // Evolve until optimal solution is found, but no longer than max_generations (to terminate at some point).
                let (best_global, found_optimal) =
                    solver.evolve_until_optimal(solution_length, max_generations);

                if is_root {
                    // Format (no newlines):
                    // exchange_generations,
                    // bechmark_repeat_time,
                    // generations,
                    // found_optimal(bool),
                    // found_solution_length
                    writeln!(
                        file,
                        "{exc},{i},{},{},{}",
                        solver.current_generation(),
                        found_optimal,
                        best_global.length()
                    )
                    .unwrap();

                    println!(
                        "{exc},{i},{},{},{}",
                        solver.current_generation(),
                        found_optimal,
                        best_global.length()
                    );
                }
            }
        }
    }
}
