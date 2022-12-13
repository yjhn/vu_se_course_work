use clap::Parser;

use crate::config;

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short, long, value_delimiter = ',')]
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
    pub benchmark_repeat_times: u32,

    #[arg(long, short, value_delimiter = ',')]
    pub algorithms: Vec<String>,
}
