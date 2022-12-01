use std::{fmt::Display, path::Path};

use mpi::topology::SystemCommunicator;
use rand::{Rng, SeedableRng};

pub fn benchmark<PD, R>(
    path: PD,
    random_seed: u64,
    world: SystemCommunicator,
    rank: i32,
    is_root: bool,
    min_processes: u32,
    max_processes: u32,
    min_generations: u32,
    max_generations: u32,
    repeat_times: u32,
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    // let mut solver = TspSolver::<config::MainRng>::from_file(
    //     &path,
    //     config::SOLUTION_STRATEGY,
    //     random_seed,
    //     world,
    // );
    // println!("Initial tour length: {}", solver.best_tour_length());

    // solver.evolve(config::EVOLUTION_GENERATION_COUNT);

    // println!(
    //     "Final tour length (for process at rank {rank}): {}",
    //     solver.best_tour_length()
    // );

    // // This must be executed by all processes.
    // let best_global = solver.best_global_tour();

    // // Root process outputs the results.
    // if is_root {
    //     println!("Best global tour length: {}", best_global.length());
    //     best_global.save_to_file(path, config::SOLUTION_FILE_NAME);
    //     println!("Best tour saved to file {}", config::SOLUTION_FILE_NAME);
    // }
}
