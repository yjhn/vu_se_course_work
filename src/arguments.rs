use clap::Parser;

use crate::config;

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short, long, required = true, num_args(1..))]
    /// TSP problem definition files.
    pub files: Vec<String>,

    #[arg(long, default_value_t = true)]
    /// True - run the benchmarks, false - find solutions to provided files.
    pub benchmark: bool,

    #[arg(short, long, default_value_t = config::benchmark::MAX_GENERATIONS)]
    /// Maximum number of generations for obtaining the optimal solution.
    pub max_generations: u32,

    #[arg(short, long, default_value_t = config::benchmark::REPEAT_TIMES)]
    /// Number of times to repeat the benchmark.
    pub bench_repeat_times: u32,

    #[arg(long, short, required = true, num_args(1..))]
    pub algorithms: Vec<String>,

    #[arg(long, default_value_t = config::benchmark::RESULTS_DIR.to_owned())]
    pub bench_results_dir: String,

    #[arg(short, long, required = false, num_args(1..))]
    pub population_sizes: Vec<u32>,

    #[arg(short, long, required = false, num_args(1..))]
    pub exchange_generations: Vec<u32>,

    #[arg(long, required = false, default_value_t = false)]
    /// Skip duplicate benchmarks if their output files are found.
    /// If this is not specified, panics if duplicates are found.
    pub skip_duplicates: bool,
}
