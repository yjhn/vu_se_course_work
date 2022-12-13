use std::{
    path::Path,
    time::{Duration, Instant},
};

use mpi::{
    collective::UserOperation,
    topology::SystemCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
};
use rand::{Rng, SeedableRng};

use crate::{
    matrix::SquareMatrix,
    probability_matrix::ProbabilityMatrix,
    timing::{GenerationsInfo, SingleGenerationTimingInfo},
    tour::{Length, Tour},
    tsp_problem::TspProblem,
    Algorithm,
};

// Position of city in all cities. Zero-based.
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
    solution_strategy: Algorithm,
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
        solution_strategy: Algorithm,
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
        solution_strategy: Algorithm,
        random_seed: u64,
        mpi: SystemCommunicator,
        exchange_generations: u32,
        population_size: u32,
    ) -> TspSolver<R> {
        let mut rng = R::seed_from_u64(random_seed);
        let mut best_tour = Tour::PLACEHOLDER;

        match solution_strategy {
            Algorithm::Cga => {
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

                // Generate population_size random tours and
                // update the prob matrix accordingly.
                for _ in 0..population_size {
                    solver.cga_generate_winner_loser::<false>();
                }
                // println!("Finished creating the initial population");

                solver
            }
            Algorithm::CgaTwoOpt | Algorithm::CgaThreeOpt => {
                let mut probability_matrix =
                    ProbabilityMatrix::new(problem.number_of_cities(), 0.0);

                // Generate POPULATION_COUNT random tours, optimize them and
                // update the prob matrix accordingly.
                for _ in 0..population_size {
                    let mut opt_tour =
                        Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);

                    match solution_strategy {
                        Algorithm::CgaTwoOpt => {
                            // opt_tour.two_opt(problem.distances());
                            opt_tour.two_opt_dlb(problem.distances());
                            // opt_tour.two_opt_take_best_each_time(problem.distances())
                        }
                        Algorithm::CgaThreeOpt => opt_tour.three_opt_dlb(problem.distances()),
                        // Algorithm::CgaThreeOpt => opt_tour.three_opt(problem.distances()),
                        Algorithm::Cga => unreachable!(),
                    }

                    probability_matrix.increase_probabilitities(&opt_tour);
                    if opt_tour.is_shorter_than(&best_tour) {
                        best_tour = opt_tour;
                    }
                    // println!("Created individual {i}");
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

    pub fn evolve(&mut self, generations: u32) -> Vec<SingleGenerationTimingInfo> {
        let mut timings = Vec::with_capacity(generations as usize);

        match self.solution_strategy {
            Algorithm::Cga => {
                for gen in 0..generations {
                    self.current_generation += 1;

                    let timing = self.evolve_inner_cga();
                    timings.push(timing);
                    if gen % self.exchange_generations == 0 {
                        self.exchange_best_tours();
                    }
                }
            }
            Algorithm::CgaTwoOpt | Algorithm::CgaThreeOpt => {
                for gen in 0..generations {
                    self.current_generation += 1;

                    let timing = self.evolve_inner_opt();
                    timings.push(timing);
                    if gen % self.exchange_generations == 0 {
                        self.exchange_best_tours();
                    }
                }
            }
        }

        timings
    }

    // Returns best global tour length, whether the best global
    // tour is optimal, and vec of best tour length for each
    // generation and generation timings.
    pub fn evolve_until_optimal(
        &mut self,
        optimal_length: u32,
        max_generations: u32,
    ) -> (u32, bool, GenerationsInfo) {
        let mut lengths_timings = GenerationsInfo::new(max_generations as usize);
        let mut best_global_len = 0;

        while self.best_tour_length() > optimal_length && self.current_generation < max_generations
        {
            self.current_generation += 1;
            let timing = match self.solution_strategy {
                Algorithm::Cga => self.evolve_inner_cga(),
                Algorithm::CgaTwoOpt | Algorithm::CgaThreeOpt => self.evolve_inner_opt(),
            };

            let best_global = if self.current_generation % self.exchange_generations == 0 {
                self.exchange_best_tours()
            } else {
                // We are not exchanging tours this generation, but still
                // collect best global tour length for statistics.
                self.best_global_tour()
            };

            best_global_len = best_global.length();
            lengths_timings.add_generation(
                timing.tour_generation_from_prob_matrix(),
                timing.tour_optimization(),
                best_global_len,
            );
            // println!("Finished generation {}", self.current_generation);

            if best_global_len == optimal_length {
                return (best_global_len, true, lengths_timings);
            } else if best_global_len < optimal_length {
                // Sanity check.
                println!(
                    "Found tour shorter than best possible:\n{:?}",
                    best_global.cities()
                );
                self.mpi.abort(2);
            }
        }

        // If this place is reached all the generations have passed,
        // their info is already in the vec and optimal tour has not been found.
        (best_global_len, false, lengths_timings)
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

    fn evolve_inner_opt(&mut self) -> SingleGenerationTimingInfo {
        let generation_timer = Instant::now();
        let loser = Tour::from_prob_matrix(
            self.number_of_cities(),
            &self.probability_matrix,
            self.problem.distances(),
            &mut self.rng,
        );
        let tour_generation_from_prob_matrix = generation_timer.elapsed();
        let mut winner = loser.clone();

        let optimization_timer = Instant::now();
        match self.solution_strategy {
            // Algorithm::CgaTwoOpt => winner.two_opt(self.distances()),
            Algorithm::CgaTwoOpt => winner.two_opt_dlb(self.distances()),
            // Algorithm::CgaTwoOpt => winner.two_opt_take_best_each_time(self.distances()),
            // Algorithm::CgaThreeOpt => winner.three_opt(self.distances()),
            Algorithm::CgaThreeOpt => winner.three_opt_dlb(self.distances()),
            Algorithm::Cga => unreachable!(),
        }
        let tour_optimization = optimization_timer.elapsed();

        // Increase probs of all paths taken by the winner and
        // decrease probs of all paths taken by the loser.
        self.probability_matrix.increase_probabilitities(&winner);
        self.probability_matrix.decrease_probabilitities(&loser);
        if winner.is_shorter_than(&self.best_tour) {
            println!(
                "New best tour length in generation {}: old {}, new {}",
                self.current_generation,
                self.best_tour.length(),
                winner.length()
            );
            self.best_tour = winner;
        }

        SingleGenerationTimingInfo::new(tour_generation_from_prob_matrix, tour_optimization)
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
                "Best tour length improved in generation {}: old {}, new {}",
                self.current_generation,
                self.best_tour.length(),
                shorter.length()
            );
            self.best_tour = shorter;
        }
    }

    fn evolve_inner_cga(&mut self) -> SingleGenerationTimingInfo {
        let gen_timer = Instant::now();
        self.cga_generate_winner_loser::<true>();
        let gen = gen_timer.elapsed();
        SingleGenerationTimingInfo::new(gen, Duration::ZERO)
    }

    pub fn best_tour_length(&self) -> u32 {
        self.best_tour.length()
    }
}
