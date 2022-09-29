use std::path::Path;

use std::ops::Range;
use tspf::{Tsp, TspBuilder};

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use crate::matrix::SquareMatrix;

// For randomly generated problems
const MIN_CITY_COORD: f64 = 0.0;
const MAX_CITY_COORD: f64 = 1000.0;

pub struct TspProblem {
    distances: SquareMatrix<f64>,
    number_of_cities: usize,
}

impl TspProblem {
    pub fn random(city_count: usize) -> TspProblem {
        let distances = generate_cities(city_count);

        TspProblem {
            distances,
            number_of_cities: city_count,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> TspProblem {
        let tsp = TspBuilder::parse_path(path).unwrap();
        let number_of_cities = tsp.dim();
        let distances = Self::calculate_distances(number_of_cities, &tsp);

        TspProblem {
            distances,
            number_of_cities,
        }
    }

    pub fn number_of_cities(&self) -> usize {
        self.number_of_cities
    }

    pub fn distances(&self) -> &SquareMatrix<f64> {
        &self.distances
    }

    fn calculate_distances(city_count: usize, tsp: &Tsp) -> SquareMatrix<f64> {
        let mut distances = SquareMatrix::new(city_count, 0.0);

        // tspf indices start from 1
        for ind1 in 1..=city_count {
            for ind2 in (ind1 + 1)..=city_count {
                let dist = tsp.weight(ind1, ind2);
                distances[(ind1 - 1, ind2 - 1)] = dist;
                distances[(ind2 - 1, ind1 - 1)] = dist;
            }
        }

        distances
    }
}

// Returns city distance matrix.
fn generate_cities(city_count: usize) -> SquareMatrix<f64> {
    let mut rng = SmallRng::from_entropy();
    let mut cities = Vec::with_capacity(city_count);

    // Generate some cities
    for _ in 0..city_count {
        let location = Point::random(&mut rng, MIN_CITY_COORD..MAX_CITY_COORD);
        cities.push(location);
    }

    // Calculate distances (Euclidean plane)
    let mut distances = SquareMatrix::new(city_count, 0.0);

    for i in 0..city_count {
        for j in (i + 1)..city_count {
            let distance = Point::distance(cities[i], cities[j]);
            distances[(i, j)] = distance;
            distances[(j, i)] = distance;
        }
    }

    distances
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn random(rng: &mut impl Rng, range: Range<f64>) -> Point {
        Point {
            x: rng.gen_range(range.clone()),
            y: rng.gen_range(range),
        }
    }

    fn distance(p1: Point, p2: Point) -> f64 {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;

        (dx * dx + dy * dy).sqrt()
    }
}
