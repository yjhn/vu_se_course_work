use std::path::Path;

use mpi::{
    collective::UserOperation,
    topology::SystemCommunicator,
    traits::{CommunicatorCollectives, Equivalence},
};
use rand::{Rng, SeedableRng};

use crate::{
    matrix::SquareMatrix,
    probability_matrix::ProbabilityMatrix,
    tour::{Length, Tour},
    tsp_problem::TspProblem,
    SolutionStrategy,
};

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
    solution_strategy: SolutionStrategy,
    // Only upper left triangle will be used.
    probability_matrix: ProbabilityMatrix,
    best_tour: Tour,
    current_generation: u32,
    exchange_generations: u32,
    rng: R,
    mpi: SystemCommunicator,
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(
        path: impl AsRef<Path>,
        solution_strategy: SolutionStrategy,
        random_seed: u64,
        mpi: SystemCommunicator,
        exchange_generations: u32,
        population_size: u32,
    ) -> TspSolver<R> {
        let problem = TspProblem::from_file(path);
        Self::from_tsp_problem(
            problem,
            solution_strategy,
            random_seed,
            mpi,
            exchange_generations,
            population_size,
        )
    }

    pub fn from_tsp_problem(
        problem: TspProblem,
        solution_strategy: SolutionStrategy,
        random_seed: u64,
        mpi: SystemCommunicator,
        exchange_generations: u32,
        population_size: u32,
    ) -> TspSolver<R> {
        let mut rng = R::seed_from_u64(random_seed);
        let mut best_tour = Tour::PLACEHOLDER;

        match solution_strategy {
            SolutionStrategy::Cga => {
                let probability_matrix = ProbabilityMatrix::new(problem.number_of_cities(), 0.5);

                let mut solver = TspSolver {
                    problem,
                    solution_strategy,
                    probability_matrix,
                    best_tour,
                    current_generation: 0,
                    exchange_generations,
                    rng,
                    mpi,
                };

                // Generate POPULATION_COUNT random tours and
                // update the prob matrix accordingly.
                for _ in 0..population_size {
                    solver.cga_generate_winner_loser::<false>();
                }

                solver
            }
            SolutionStrategy::CgaTwoOpt | SolutionStrategy::CgaThreeOpt => {
                let mut probability_matrix =
                    ProbabilityMatrix::new(problem.number_of_cities(), 0.0);

                // Generate POPULATION_COUNT random tours, optimize them and
                // update the prob matrix accordingly.
                for i in 0..population_size {
                    let mut opt_tour =
                        Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);

                    match solution_strategy {
                        SolutionStrategy::CgaTwoOpt => {
                            opt_tour.two_opt_take_best_each_time(problem.distances())
                        }
                        SolutionStrategy::CgaThreeOpt => opt_tour.three_opt(problem.distances()),
                        SolutionStrategy::Cga => unreachable!(),
                    }

                    probability_matrix.increase_probabilitities(&opt_tour);

                    if opt_tour.is_shorter_than(&best_tour) {
                        best_tour = opt_tour;
                    }
                    println!("Created population member {i}");
                }
                println!("Finished creating the initial population");

                TspSolver {
                    problem,
                    solution_strategy,
                    probability_matrix,
                    best_tour,
                    current_generation: 0,
                    exchange_generations,
                    rng,
                    mpi,
                }
            }
        }
    }

    pub fn number_of_cities(&self) -> usize {
        self.problem.number_of_cities()
    }

    pub fn distances(&self) -> &SquareMatrix<u32> {
        self.problem.distances()
    }

    pub fn current_generation(&self) -> u32 {
        self.current_generation
    }

    pub fn evolve(&mut self, generations: u32) {
        match self.solution_strategy {
            SolutionStrategy::Cga => {
                for gen in 0..generations {
                    self.current_generation += 1;

                    self.evolve_inner_cga();
                    if gen % self.exchange_generations == 0 {
                        self.exchange_best_tours();
                    }
                }
            }
            SolutionStrategy::CgaTwoOpt | SolutionStrategy::CgaThreeOpt => {
                for gen in 0..generations {
                    self.current_generation += 1;

                    self.evolve_inner_opt();
                    if gen % self.exchange_generations == 0 {
                        self.exchange_best_tours();
                    }
                }
            }
        }
    }

    // Returns best global tour adn whether it is optimal.
    pub fn evolve_until_optimal(
        &mut self,
        optimal_length: u32,
        max_generations: u32,
    ) -> (Tour, bool) {
        match self.solution_strategy {
            SolutionStrategy::Cga => {
                while self.best_tour_length() > optimal_length
                    && self.current_generation < max_generations
                {
                    self.current_generation += 1;

                    self.evolve_inner_cga();
                    if self.current_generation % self.exchange_generations == 0 {
                        let best_global = self.exchange_best_tours();
                        if best_global.length() == optimal_length {
                            return (best_global, true);
                        }
                    }
                }
            }
            SolutionStrategy::CgaTwoOpt | SolutionStrategy::CgaThreeOpt => {
                while self.best_tour_length() > optimal_length
                    && self.current_generation < max_generations
                {
                    self.current_generation += 1;

                    self.evolve_inner_opt();
                    if self.current_generation % self.exchange_generations == 0 {
                        let best_global = self.exchange_best_tours();
                        if best_global.length() == optimal_length {
                            return (best_global, true);
                        }
                    }
                }
            }
        }

        let best_global = self.best_global_tour();
        let best_global_length = best_global.length();
        (best_global, best_global_length == optimal_length)
    }

    pub fn problem_name(&self) -> &str {
        self.problem.name()
    }

    // Returns the best global tour.
    fn exchange_best_tours(&mut self) -> Tour {
        let global_best = self.best_global_tour();

        // Update the probability matrix if the global best tour is
        // shorter than local best tour (one process will have its tour
        // chosen as the global best tour).
        if global_best.is_shorter_than(&self.best_tour) {
            self.probability_matrix
                .decrease_probabilitities(&self.best_tour);
            self.probability_matrix
                .increase_probabilitities(&global_best);
        }

        global_best
    }

    pub fn best_global_tour(&mut self) -> Tour {
        // Exchange best tours.
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

    fn evolve_inner_opt(&mut self) {
        let loser = Tour::from_prob_matrix(
            self.number_of_cities(),
            &self.probability_matrix,
            self.problem.distances(),
            &mut self.rng,
        );
        let mut winner = loser.clone();

        match self.solution_strategy {
            SolutionStrategy::CgaTwoOpt => winner.two_opt_take_best_each_time(self.distances()),
            SolutionStrategy::CgaThreeOpt => winner.three_opt(self.distances()),
            SolutionStrategy::Cga => unreachable!(),
        }

        // Increase probs of all paths taken by the winner and
        // decrease probs of all paths taken by the loser.
        self.probability_matrix.increase_probabilitities(&winner);
        self.probability_matrix.decrease_probabilitities(&loser);
        if winner.is_shorter_than(&self.best_tour) {
            self.best_tour = winner;
            println!(
                "New best tour length in generation {}: {}",
                self.current_generation,
                self.best_tour.length()
            );
        }
    }

    fn cga_generate_winner_loser<const FROM_PROBS: bool>(&mut self) {
        let (tour_a, tour_b) = if FROM_PROBS {
            let tour_a = Tour::from_prob_matrix(
                self.number_of_cities(),
                &self.probability_matrix,
                self.problem.distances(),
                &mut self.rng,
            );
            let tour_b = Tour::from_prob_matrix(
                self.number_of_cities(),
                &self.probability_matrix,
                self.problem.distances(),
                &mut self.rng,
            );
            (tour_a, tour_b)
        } else {
            let tour_a = Tour::random(
                self.number_of_cities(),
                self.problem.distances(),
                &mut self.rng,
            );
            let tour_b = Tour::random(
                self.number_of_cities(),
                self.problem.distances(),
                &mut self.rng,
            );
            (tour_a, tour_b)
        };

        let (shorter, longer) = if tour_a.is_shorter_than(&tour_b) {
            (tour_a, tour_b)
        } else {
            (tour_b, tour_a)
        };
        self.probability_matrix.increase_probabilitities(&shorter);
        self.probability_matrix.decrease_probabilitities(&longer);

        if shorter.is_shorter_than(&self.best_tour) {
            println!(
                "Best tour length improved in generation {}, old {}, new {}",
                self.current_generation,
                self.best_tour.length(),
                shorter.length()
            );
            self.best_tour = shorter;
        }
    }

    fn evolve_inner_cga(&mut self) {
        self.cga_generate_winner_loser::<true>();
    }

    pub fn best_tour_length(&self) -> u32 {
        self.best_tour.length()
    }
}
