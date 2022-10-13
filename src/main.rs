mod bench;
mod matrix;
mod tour;
mod tsp_problem;

use std::{env, path::Path};

use matrix::SquareMatrix;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tour::{Tour, TourIndex};
use tsp_problem::TspProblem;

const TEST_FILE: &str = "test_data/a10.tsp";
const EVOLUTION_GENERATION_COUNT: u32 = 10;
const POPULATION_COUNT: u32 = 128;
const INCREMENT: f64 = 1_f64 / POPULATION_COUNT as f64;

fn main() {
    let path = get_path().unwrap_or_else(|| TEST_FILE.to_owned());
    println!("File path: {path}");
    let random_seed = rand::random();

    let mut solver = TspSolver::<SmallRng>::from_file(path, random_seed);
    println!("Initial tour length: {}", solver.best_tour.length());

    solver.evolve(EVOLUTION_GENERATION_COUNT);
}

fn get_path() -> Option<String> {
    let mut args = env::args();
    // First arg is usually program path or empty.
    args.next();
    args.next()
}

// Position of city in all cities. Zero-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(path: impl AsRef<Path>, random_seed: u64) -> TspSolver<R> {
        let problem = TspProblem::from_file(path);
        Self::from_tsp_problem(problem, random_seed)
    }

    pub fn from_tsp_problem(problem: TspProblem, random_seed: u64) -> TspSolver<R> {
        let mut probability_matrix = SquareMatrix::new(problem.number_of_cities(), 0.0);
        let mut rng = R::seed_from_u64(random_seed);
        let mut best_tour = Tour::PLACEHOLDER;

        // Generate POPULATION_COUNT random cities, optimize them and
        // update the prob matrix accordingly.
        for _ in 0..POPULATION_COUNT {
            let mut opt_tour =
                Tour::random(problem.number_of_cities(), problem.distances(), &mut rng);
            // println!("reached line: {}", line!());

            opt_tour.ls_2_opt_take_best(problem.distances());
            // println!("reached line: {}", line!());

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
        for _ in 0..generations {
            self.current_generation += 1;

            let loser = self.gen_tour_from_prob_matrix();
            let mut winner = loser.clone();
            // println!("reached line {} in {}", line!(), file!());

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
            // println!("reached line {} in {}", line!(), file!());

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

    // Kinda useless, since this is not used in the paper it seems.
    // pub fn update_probabilities(&mut self, t1: &Tour, t2: &Tour) {
    //     // Update each path to have higher probability if
    //     // the winner has it and lower otherwise.
    //     if t1.is_shorter_than(t2) {
    //         self.update_probabilitities::<true>(t1);
    //         self.update_probabilitities::<false>(t2);
    //     } else {
    //         self.update_probabilitities::<true>(t2);
    //         self.update_probabilitities::<false>(t1);
    //     }
    // }

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

/*
pub struct TwoOptTspSolver {
    problem: TspProblem,
    tour: Tour,
}

impl TwoOptTspSolver {
    pub fn new(tsp_problem: TspProblem, random_seed: u64) -> TwoOptTspSolver {
        // Construct a tour using nearest-neighbour algorithm.
        todo!()
    }
}

trait TspSolve {
    fn new(tsp_problem: TspProblem, random_seed: u64) -> Self;

    fn solve(&mut self) -> Tour;
}
*/
