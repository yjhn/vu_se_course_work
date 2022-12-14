use std::fs::{self, OpenOptions};
use std::io::Write;
use std::{fmt::Display, path::Path};

use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};

use crate::initialize_random_seed;
use crate::tsp_problem::TspProblem;
use crate::{tsp_solver::TspSolver, Algorithm};

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
    population_sizes: &[u32],
    exchange_generations: &[u32],
    algorithms: &[Algorithm],
    skip_duplicates: bool,
    results_dir: &str,
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    if is_root {
        // Create benchmark results directory.
        if !Path::new(results_dir).exists() {
            fs::create_dir(results_dir).unwrap();
        }

        println!("Problem file name: {path}");
        println!("MPI processes: {}", world.size());
        println!("Problem name: {problem_name}");
        println!("Population size: {population_sizes:?}");
    }

    for &p in population_sizes {
        for &a in algorithms {
            let problem = TspProblem::from_file(&path);
            let save_file_path = format!(
                "{results_dir}/bm_{problem_name}_{a}_{}_cpus_p{p}.out",
                world.size()
            );

            // This must be an array upfront, we can't make this an array
            // at root_process.broadcast_into() call site for some reason.
            // TODO: investigate.
            let mut skip = [false];
            let mut file = if is_root {
                let res_file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&save_file_path);

                match res_file {
                    Ok(f) => f,
                    Err(_) => {
                        if skip_duplicates {
                            println!(
                                "Skipping benchmark due to existing results file:\n{save_file_path}"
                            );
                            skip[0] = true;

                            OpenOptions::new()
                                .read(false)
                                .write(true)
                                .open("/dev/null")
                                .unwrap()
                        } else {
                            println!(
                                "Will not overwrite existing benchmark results file:\n{save_file_path}"
                            );
                            world.abort(1);
                        }
                    }
                }
            } else {
                // Blackhole writes from other threads.
                OpenOptions::new()
                    .read(false)
                    .write(true)
                    .open("/dev/null")
                    .unwrap()
            };
            // Exchange info whether this iteration needs to be skipped.
            root_process.broadcast_into(&mut skip);
            if skip[0] {
                continue;
            }

            if is_root {
                writeln!(file, "Problem file name: {path}").unwrap();
                writeln!(file, "Problem name: {problem_name}").unwrap();
                writeln!(file, "MPI processes: {}", world.size()).unwrap();
                writeln!(file, "Population size: {p}\n\n\n\n\n").unwrap();
            }

            // \n\n = record separator
            // \n\n\n = exchange generations separator
            // \n\n\n\n = algorithm separator
            // \n\n\n\n\n\n = file header separator
            if is_root {
                println!("Algorithm used: {a}");
            }

            for exc in exchange_generations {
                for i in 0..repeat_times {
                    // Separate random seed must be created for each run.
                    let random_seed = initialize_random_seed(root_process, rank, is_root);

                    let mut solver = TspSolver::<R>::from_tsp_problem(
                        problem.clone(),
                        a,
                        random_seed,
                        world,
                        *exc,
                        p,
                    );

                    // Evolve until optimal solution is found, but no longer than max_generations (to terminate at some point).
                    let (
                        best_global_len,
                        found_optimal,
                        generations_info,
                        avg_tour_exchange_duration,
                    ) = solver.evolve_until_optimal(solution_length, max_generations);

                    if is_root {
                        // Format (no newlines except \n):
                        // algorithm,
                        // exchange_generations,
                        // bechmark_repeat_time,
                        // generations,
                        // found_optimal(bool),
                        // found_solution_length
                        // \n
                        // for each generation:
                        // best_global_len
                        // \n
                        // tour generation time
                        // \n
                        // tour optimization time
                        write!(
                            file,
                            "{a},{exc},{i},{},{},{},{}\n",
                            solver.current_generation(),
                            found_optimal,
                            best_global_len,
                            avg_tour_exchange_duration.as_micros()
                        )
                        .unwrap();
                        // Generations info.
                        for l in generations_info.optimized_length() {
                            write!(file, "{l},").unwrap();
                        }
                        writeln!(file).unwrap();
                        for t_gen in generations_info.tour_generation_from_prob_matrix() {
                            // Milliseconds are not high enough resolution,
                            // are always zero for tour generation.
                            write!(file, "{},", t_gen.as_micros()).unwrap();
                        }
                        writeln!(file).unwrap();
                        for t_opt in generations_info.tour_optimization() {
                            write!(file, "{},", t_opt.as_micros()).unwrap();
                        }
                        write!(file, "\n\n").unwrap();

                        println!(
                            "{exc},{i},{},{},{}",
                            solver.current_generation(),
                            found_optimal,
                            best_global_len
                        );
                    }
                }

                write!(file, "\n\n\n").unwrap();
            }
        }
    }
}
