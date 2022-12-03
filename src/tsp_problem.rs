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
    name: String,
    cities: Vec<Point>,
    distances: SquareMatrix<u32>,
}

impl TspProblem {
    pub fn random(city_count: usize) -> TspProblem {
        let (cities, distances) = generate_cities(city_count);

        TspProblem {
            name: String::from("random"),
            cities,
            distances,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> TspProblem {
        let tsp = TspBuilder::parse_path(path).unwrap();
        let number_of_cities = tsp.dim();
        let cities = {
            let coord_map = tsp.node_coords();
            let mut cities = Vec::with_capacity(number_of_cities);
            for idx in 1..=number_of_cities {
                let tspf_point_pos = coord_map.get(&idx).unwrap().pos();

                // We only care about 2D.
                let point = Point::new(tspf_point_pos[0], tspf_point_pos[1]);

                cities.push(point);
            }

            cities
        };

        let distances = Self::calculate_distances(number_of_cities, &tsp);

        TspProblem {
            name: tsp.name().to_owned(),
            cities,
            distances,
        }
    }

    pub fn number_of_cities(&self) -> usize {
        self.cities.len()
    }

    pub fn distances(&self) -> &SquareMatrix<u32> {
        &self.distances
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn calculate_distances(city_count: usize, tsp: &Tsp) -> SquareMatrix<u32> {
        let mut distances = SquareMatrix::new(city_count, 0);

        // tspf indices start from 1
        for ind1 in 1..=city_count {
            for ind2 in (ind1 + 1)..=city_count {
                let dist_f64 = tsp.weight(ind1, ind2);
                let dist = nint(dist_f64);
                distances[(ind1 - 1, ind2 - 1)] = dist;
                distances[(ind2 - 1, ind1 - 1)] = dist;
            }
        }

        distances
    }
}

// Returns city distance matrix.
fn generate_cities(city_count: usize) -> (Vec<Point>, SquareMatrix<u32>) {
    let mut rng = SmallRng::from_entropy();
    let mut cities = Vec::with_capacity(city_count);

    // Generate some cities
    for _ in 0..city_count {
        let location = Point::random(&mut rng, MIN_CITY_COORD..MAX_CITY_COORD);
        cities.push(location);
    }

    // Calculate distances (Euclidean plane)
    let mut distances = SquareMatrix::new(city_count, 0);

    for i in 0..city_count {
        for j in (i + 1)..city_count {
            let distance_f64 = Point::distance(cities[i], cities[j]);
            let distance = nint(distance_f64);
            distances[(i, j)] = distance;
            distances[(j, i)] = distance;
        }
    }

    (cities, distances)
}

// Rounding the same way as nint() function defined in TSPLIB95 format.
fn nint(f: f64) -> u32 {
    (f + 0.5).round() as u32
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    pub fn random(rng: &mut impl Rng, range: Range<f64>) -> Point {
        Point {
            x: rng.gen_range(range.clone()),
            y: rng.gen_range(range),
        }
    }

    pub fn distance(p1: Point, p2: Point) -> f64 {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;

        (dx * dx + dy * dy).sqrt()
    }
}
