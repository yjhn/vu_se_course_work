use clap::Parser;

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short, long, value_delimiter = ',')]
    /// TSP problem definition files.
    pub files: Vec<String>,

    #[arg(short, long, default_value_t = true)]
    /// True - run the benchmarks, false - find solutions to provided files.
    pub benchmark: bool,

    #[arg(short, long, default_value_t = 500)]
    /// Maximum number of generations for obtaining the optimal solution.
    pub max_generations: u32,
}
