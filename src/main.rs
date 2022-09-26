use std::{collections::HashMap, path::Path};

use matrix::SquareMatrix;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tour::Tour;
use tspf::{Point, Tsp, TspBuilder};

const TEST_FILE: &str = "test_data/a280.tsp";
const RANDOM_SEED: u64 = 1543434354;
const EVOLUTION_GENERATION_COUNT: u32 = 10;
const POPULATION_COUNT: u32 = 100;

fn main() {
    let mut solver = TspSolver::<SmallRng>::from_file(TEST_FILE, RANDOM_SEED);
    solver.random_as_best();
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

// Position of city in the tour. Zero-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TourIndex(usize);

impl TourIndex {
    pub fn new(index: usize) -> TourIndex {
        TourIndex(index)
    }

    pub fn get(&self) -> usize {
        self.0
    }

    pub fn inc(&mut self, size: usize) -> TourIndex {
        self.0 = (self.0 + 1) % size;
        *self
    }
}

mod tour {
    use crate::{matrix::SquareMatrix, CityIndex, TourIndex};

    pub struct Tour {
        cities: Vec<CityIndex>,
        tour_length: f64,
    }

    impl Tour {
        pub const PLACEHOLDER: Tour = Tour {
            cities: Vec::new(),
            tour_length: f64::INFINITY,
        };

        pub fn from_cities(cities: Vec<CityIndex>, distances: &SquareMatrix<f64>) -> Tour {
            // No empty tours allowed.
            assert!(cities.len() > 0);

            let mut tour_length = 0.0;
            for idx in 0..(cities.len() - 1) {
                tour_length += distances[(cities[idx].get(), cities[idx + 1].get())];
            }
            // Add distance from last to first.
            tour_length += distances[(cities.last().unwrap().get(), cities[0].get())];

            Tour {
                cities,
                tour_length,
            }
        }

        // From https://tsp-basics.blogspot.com/2017/02/building-blocks-reversing-segment.html
        pub fn reverse_segment(&mut self, start: TourIndex, end: TourIndex) {
            let mut left = start.get();
            let mut right = end.get();
            let len = self.cities.len();

            let inversion_size = ((len + end.get() - start.get() + 1) % len) / 2;

            for _ in 1..inversion_size {
                self.cities.swap(left, right);
                left = (left + 1) % len;
                right = (len + right - 1) % len;
            }
        }

        // More is better.
        pub fn gain_from_2_opt(
            x1: CityIndex,
            x2: CityIndex,
            y1: CityIndex,
            y2: CityIndex,
            distances: &SquareMatrix<f64>,
        ) -> f64 {
            let del_length = distances[(x1.get(), x2.get())] + distances[(y1.get(), y2.get())];
            let add_length = distances[(x1.get(), y1.get())] + distances[(x2.get(), y2.get())];

            del_length - add_length
        }

        // From https://tsp-basics.blogspot.com/2017/03/2-opt-move.html
        pub fn make_2_opt_move(&mut self, mut i: TourIndex, j: TourIndex) {
            self.reverse_segment(i.inc(self.cities.len()), j);
        }
    }
}

pub struct TspSolver<R: Rng + SeedableRng> {
    distances: SquareMatrix<f64>,
    probability_matrix: SquareMatrix<f64>,
    number_of_cities: usize,
    best_tour: Tour,
    rng: R,
}

impl<R: Rng + SeedableRng> TspSolver<R> {
    pub fn from_file(path: impl AsRef<Path>, random_seed: u64) -> TspSolver<R> {
        let tsp = TspBuilder::parse_path(path).unwrap();

        let number_of_cities = tsp.dim();
        let cities = tsp.node_coords();
        let distances = Self::calculate_distances(cities, &tsp);
        let probability_matrix = SquareMatrix::new(number_of_cities, 0.5);

        TspSolver {
            distances,
            probability_matrix,
            number_of_cities,
            best_tour: Tour::PLACEHOLDER,
            rng: R::seed_from_u64(random_seed),
        }
    }

    pub fn random_as_best(&mut self) {
        self.best_tour = self.random_tour();
    }

    // This is O(n^2), but I don't know how to optimize it.
    pub fn random_tour(&mut self) -> Tour {
        let mut cities = Vec::with_capacity(self.number_of_cities);
        let start = self.rng.gen_range(0..self.number_of_cities);
        cities.push(CityIndex::new(start));
        for idx in 1..self.number_of_cities {
            // Generate indices in unused cities only to avoid duplicates.
            let index = self.rng.gen_range(0..(self.number_of_cities - idx));
            // Check for unused cities and choose index-th unused city.
            let mut unused_city_count = 0;
            for c in 0..self.number_of_cities {
                let city_index = CityIndex(c);
                if !cities.contains(&city_index) {
                    if unused_city_count == index {
                        cities.push(city_index);
                        break;
                    }
                    unused_city_count += 1;
                }
            }
        }

        Tour::from_cities(cities, &self.distances)
    }

    pub fn optimize_tour(&self, tour: &mut Tour) {
        //TODO: add LK limited to 2-opt
    }

    pub fn evolve(&mut self, generations: u32) {
        //TODO: evolution
    }

    fn calculate_distances(cities: &HashMap<usize, Point>, tsp: &Tsp) -> SquareMatrix<f64> {
        let city_count = cities.len();
        let mut distances = SquareMatrix::new(city_count, 0.0);

        for ind1 in 0..city_count {
            for ind2 in (ind1 + 1)..city_count {
                let dist = tsp.weight(ind1 + 1, ind2);
                distances[(ind1, ind2)] = dist;
                distances[(ind2, ind1)] = dist;
            }
        }

        distances
    }
}

mod matrix {
    use std::ops::{Index, IndexMut};

    pub struct SquareMatrix<T>
    where
        T: Copy,
    {
        data: Vec<T>,
        side_length: usize,
    }

    impl<T> SquareMatrix<T>
    where
        T: Copy,
    {
        pub fn new(side_length: usize, init_value: T) -> SquareMatrix<T> {
            let mut data = vec![init_value; side_length * side_length];

            SquareMatrix { data, side_length }
        }
    }

    impl<T> Index<(usize, usize)> for SquareMatrix<T>
    where
        T: Copy,
    {
        type Output = T;

        fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
            &self.data[self.side_length * y + x]
        }
    }

    impl<T> IndexMut<(usize, usize)> for SquareMatrix<T>
    where
        T: Copy,
    {
        fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
            &mut self.data[self.side_length * y + x]
        }
    }
}
