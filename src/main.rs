mod bench;
mod matrix;
mod tour;
mod tsp_problem;

use std::{env, path::Path};

use matrix::SquareMatrix;
use mpi::{
    collective::UserOperation,
    topology::SystemCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tour::{Length, Tour, TourIndex};
use tsp_problem::TspProblem;

const TEST_FILE: &str = "test_data/a10.tsp";
const EVOLUTION_GENERATION_COUNT: u32 = 10;
const POPULATION_COUNT: u32 = 128;
const INCREMENT: f64 = 1_f64 / POPULATION_COUNT as f64;
const EXCHANGE_GENERATIONS: u32 = 4;

const GLOBAL_SEED: u64 = 865376825679;

// Build and run:
// cargo build --release && RUST_BACKTRACE=1  mpirun --mca opal_warn_on_missing_libcuda 0 target/release/salesman test_data/a280.tsp
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    println!("World size: {size}");

    let path = get_path().unwrap_or_else(|| TEST_FILE.to_owned());
    println!("File path: {path}");
    let random_seed = GLOBAL_SEED + rank as u64; //rand::random();
    println!("Random seed: {random_seed}");

    let mut solver = TspSolver::<SmallRng>::from_file(path, random_seed, world);
    println!("Initial tour length: {}", solver.best_tour.length());

    solver.evolve(EVOLUTION_GENERATION_COUNT);

    println!("Final tour length: {}", solver.best_tour.length());
}

fn get_path() -> Option<String> {
    let mut args = env::args();
    // First arg is usually program path or empty.
    args.next();
    args.next()
}

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
    // Only upper left triangle will be used.
    probability_matrix: SquareMatrix<f64>,
    best_tour: Tour,
    current_generation: u32,
    rng: R,
    mpi: SystemCommunicator,
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(
        path: impl AsRef<Path>,
        random_seed: u64,
        mpi: SystemCommunicator,
    ) -> TspSolver<R> {
        let problem = TspProblem::from_file(path);
        Self::from_tsp_problem(problem, random_seed, mpi)
    }

    pub fn from_tsp_problem(
        problem: TspProblem,
        random_seed: u64,
        mpi: SystemCommunicator,
    ) -> TspSolver<R> {
        let mut probability_matrix = SquareMatrix::new(problem.number_of_cities(), 0.0);
        let mut rng = R::seed_from_u64(random_seed);
        let mut best_tour = Tour::PLACEHOLDER;

        // Generate POPULATION_COUNT random tours, optimize them and
        // update the prob matrix accordingly.
        for _ in 0..POPULATION_COUNT {
            let mut opt_tour =
                Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);

            opt_tour.ls_2_opt_take_best(problem.distances());

            Self::update_probabilitities::<true>(&mut probability_matrix, &opt_tour);

            if opt_tour.is_shorter_than(&best_tour) {
                best_tour = opt_tour;
            }
        }

        TspSolver {
            problem,
            probability_matrix,
            best_tour,
            current_generation: 0,
            rng,
            mpi,
        }
    }

    pub fn random_as_best(&mut self) {
        self.best_tour = Tour::random(
            self.number_of_cities(),
            self.problem.distances(),
            &mut self.rng,
        );
    }

    pub fn number_of_cities(&self) -> usize {
        self.problem.number_of_cities()
    }

    pub fn distances(&self) -> &SquareMatrix<f64> {
        self.problem.distances()
    }

    pub fn evolve(&mut self, generations: u32) {
        for gen in 0..generations {
            self.current_generation += 1;

            self.evolve_inner();
            if gen % EXCHANGE_GENERATIONS == 0 {
                self.exchange_best_tours();
            }
        }
    }

    fn exchange_best_tours(&mut self) {
        let global_best_vec = self.exchange_inner();
        assert_eq!(global_best_vec.len(), self.best_tour.city_count());

        let global_best = Tour::from_cities(global_best_vec, self.distances());

        // Update the probability matrix if the global best tour is
        // shorter than local best tour (one process will have its tour
        // chosen as the global best tour).
        if global_best.is_shorter_than(&self.best_tour) {
            Self::update_probabilitities::<false>(&mut self.probability_matrix, &self.best_tour);
            Self::update_probabilitities::<true>(&mut self.probability_matrix, &global_best);
        }

        // TODO: maybe it's worth it to also update our local best tour?
        // The paper doesn't do it for some reason.
    }

    fn exchange_inner(&mut self) -> Vec<CityIndex> {
        // Exchange best tours.
        // for now don't exchange tour length
        // length could be transmuted into usize and added at the end
        let mpi_closure = UserOperation::commutative(|read_buf, write_buf| {
            println!("Exchanging global tours");

            let local_best = read_buf.downcast::<CityIndex>().unwrap();
            let global_best = write_buf.downcast::<CityIndex>().unwrap();

            // TODO: use hack with tour length transmuting to usize to speed this up.
            let local_tour_length = local_best.calculate_tour_length(&mut self.problem.distances());
            let global_tour_length = global_best.calculate_tour_length(&self.problem.distances());
            println!("GBTL: {global_tour_length}, LBTL: {local_tour_length}");

            if local_tour_length < global_tour_length {
                global_best.copy_from_slice(local_best);
            }
        });

        // This needs to be filled up upfront.
        let mut best_global_tour: Vec<CityIndex> = self.best_tour.cities().to_owned();

        self.mpi
            .all_reduce_into(self.best_tour.cities(), &mut best_global_tour, &mpi_closure);

        best_global_tour
    }

    fn evolve_inner(&mut self) {
        let loser = self.gen_tour_from_prob_matrix();
        let mut winner = loser.clone();

        winner.ls_2_opt_take_best(self.distances());

        // Increase probs of all paths taken by the winner and
        // decrease probs of all paths taken by the loser.
        Self::update_probabilitities::<true>(&mut self.probability_matrix, &winner);
        Self::update_probabilitities::<false>(&mut self.probability_matrix, &loser);
        if winner.is_shorter_than(&self.best_tour) {
            self.best_tour = winner;
            println!(
                "New best tour length in generation {}: {}",
                self.current_generation,
                self.best_tour.length()
            );
        } else {
            println!(
                "Generation {} did not improve best tour, winner length: {}",
                self.current_generation,
                winner.length()
            );
        }
    }

    // TODO: maybe use something like C++ std::shuffle(). It could be much faster.
    fn gen_tour_from_prob_matrix(&mut self) -> Tour {
        let city_count = self.number_of_cities();
        let mut cities = Vec::with_capacity(city_count);

        let starting_city = self.rng.gen_range(0..city_count);
        cities.push(CityIndex::new(starting_city));
        let mut cities_left = city_count - 1;

        // println!("reached line {} in {}", line!(), file!());

        'outermost: while cities_left > 0 {
            let last: usize = cities.last().unwrap().get();

            // Allow trying to insert the city `cities_left` times, then,
            // if still unsuccessful, insert the city with highest probability.
            for _ in 0..cities_left {
                // Generate indices in unused cities only to avoid duplicates.
                let index = self.rng.gen_range(0..cities_left);
                let prob = self.rng.gen::<f64>();

                // Check for unused cities and choose index-th unused city.
                let mut unused_city_count = 0;
                for c in 0..city_count {
                    let city_index = CityIndex(c);
                    if !cities.contains(&city_index) {
                        if unused_city_count == index {
                            let (l, h) = Self::order(last, c);
                            if prob <= self.probability_matrix[(h, l)] {
                                cities.push(city_index);
                                // This causes false-positive warning #[warn(clippy::mut_range_bound)]
                                cities_left -= 1;
                                continue 'outermost;
                            }

                            break;
                        }
                        unused_city_count += 1;
                    }
                }
            }
            // If the control flow reaches here, insert city with highest prob.
            let (mut max_prob, mut max_prob_city, mut max_prob_dist) = (0.0, 0, f64::INFINITY);
            for _ in 0..cities_left {
                for c in 0..city_count {
                    let city_index = CityIndex(c);
                    if !cities.contains(&city_index) {
                        let (l, h) = Self::order(last, c);
                        let prob = self.probability_matrix[(h, l)];
                        if prob > max_prob {
                            let dist = self.distance(h, l);
                            (max_prob, max_prob_city, max_prob_dist) = (prob, c, dist);
                        } else if prob == max_prob {
                            let dist = self.distance(h, l);
                            if dist < max_prob_dist {
                                (max_prob, max_prob_city, max_prob_dist) = (prob, c, dist);
                            }
                        }
                    }
                }
            }
            cities.push(CityIndex::new(max_prob_city));
            cities_left -= 1;
        }

        Tour::from_cities(cities, self.distances())
    }

    fn distance(&self, a: usize, b: usize) -> f64 {
        self.problem.distances()[(a, b)]
    }

    // Returns (low, high).
    fn order(a: usize, b: usize) -> (usize, usize) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn update_probabilitities<const INC: bool>(prob_matrix: &mut SquareMatrix<f64>, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                let (l, h) = Self::order(c1.get(), c2.get());
                if INC {
                    prob_matrix[(h, l)] += INCREMENT;
                } else {
                    prob_matrix[(h, l)] -= INCREMENT;
                }
                // All values in probability matrix must always be in range [0..1].
                prob_matrix[(h, l)] = f64::clamp(prob_matrix[(h, l)], 0.0, 1.0)
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.get_path(TourIndex::new(0), TourIndex::new(t.city_count() - 1));
        let (l, h) = Self::order(c1.get(), c2.get());

        if INC {
            prob_matrix[(h, l)] += INCREMENT;
        } else {
            prob_matrix[(h, l)] -= INCREMENT;
        }
        // All values in probability matrix must always be in range [0..1].
        prob_matrix[(h, l)] = f64::clamp(prob_matrix[(h, l)], 0.0, 1.0)
    }
}
