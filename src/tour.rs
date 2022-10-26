use std::fmt::Display;
use std::io::Write;
use std::{fs::File, io::BufWriter, path::Path, slice::Windows};

use rand::seq::SliceRandom;
use rand::Rng;

use crate::{matrix::SquareMatrix, CityIndex};

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

    pub fn inc_mod(&mut self, max_size: usize) -> TourIndex {
        self.0 = (self.0 + 1) % max_size;
        *self
    }
}

#[derive(Clone, Debug)]
pub struct Tour {
    cities: Vec<CityIndex>,
    tour_length: f64,
}

impl Tour {
    pub const PLACEHOLDER: Tour = Tour {
        cities: Vec::new(),
        tour_length: f64::INFINITY,
    };

    pub fn city_count(&self) -> usize {
        self.cities.len()
    }

    pub fn length(&self) -> f64 {
        self.tour_length
    }

    pub fn from_cities(cities: Vec<CityIndex>, distances: &SquareMatrix<f64>) -> Tour {
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    pub fn random(city_count: usize, distances: &SquareMatrix<f64>, rng: &mut impl Rng) -> Tour {
        assert!(city_count > 1);

        let mut cities: Vec<CityIndex> = (0..city_count).map(|c| CityIndex::new(c)).collect();
        cities.shuffle(rng);
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    // From https://tsp-basics.blogspot.com/2017/02/building-blocks-reversing-segment.html
    fn reverse_segment(&mut self, start: TourIndex, end: TourIndex) {
        let mut left = start.get();
        let mut right = end.get();
        let len = self.cities.len();

        let inversion_size = ((len + right - left + 1) % len) / 2;
        assert!(inversion_size >= 1);

        for _ in 1..=inversion_size {
            self.cities.swap(left, right);
            left = (left + 1) % len;
            right = (len + right - 1) % len;
        }
    }

    // More is better.
    fn gain_from_2_opt(
        x1: CityIndex,
        x2: CityIndex,
        y1: CityIndex,
        y2: CityIndex,
        distances: &SquareMatrix<f64>,
    ) -> f64 {
        let (x1, x2, y1, y2) = (x1.get(), x2.get(), y1.get(), y2.get());

        let del_length = distances[(x1, x2)] + distances[(y1, y2)];
        let add_length = distances[(x1, y1)] + distances[(x2, y2)];

        del_length - add_length
    }

    // From https://tsp-basics.blogspot.com/2017/03/2-opt-move.html
    // a..b in Nim is inclusive!!!
    // https://nim-lang.org/docs/tut1.html#control-flow-statements-for-statement
    fn make_2_opt_move(&mut self, mut i: TourIndex, j: TourIndex, move_gain: f64) {
        self.reverse_segment(i.inc_mod(self.cities.len()), j);
        // This is not perfectly accurate, but will be good enough.
        self.tour_length -= move_gain;
    }

    /*
    pub fn ls_2_opt(&mut self, distances: &SquareMatrix<f64>) {
        let mut locally_optimal = false;
        let len = self.cities.len();

        while !locally_optimal {
            locally_optimal = true;
            'outer: for counter_1 in 0..(len - 2) {
                let i = counter_1;
                let x1 = self.cities[i];
                let x2 = self.cities[(i + 1) % len];

                let counter_2_limit = if i == 0 { len - 1 } else { len };

                for counter_2 in (i + 2)..counter_2_limit {
                    let j = counter_2;
                    let y1 = self.cities[j];
                    let y2 = self.cities[(j + 1) % len];

                    if Self::gain_from_2_opt(x1, x2, y1, y2, distances) > 0.0 {
                        self.make_2_opt_move(TourIndex::new(i), TourIndex::new(j));
                        locally_optimal = false;
                        break 'outer;
                    }
                }
            }
        }
    }*/

    // Make 2-opt moves until no improvements can be made.
    // Choose the best possible move each time.
    pub fn two_opt_take_best_each_time(&mut self, distances: &SquareMatrix<f64>) {
        #[derive(Debug, Clone, Copy)]
        struct TwoOptMove {
            i: TourIndex,
            j: TourIndex,
        }

        let len = self.cities.len();

        loop {
            let mut best_move_gain = 0.0;
            // There might not be any moves that shorten the tour.
            let mut best_move: Option<TwoOptMove> = None;

            for i in 0..(len - 2) {
                let x1 = self.cities[i];
                let x2 = self.cities[(i + 1) % len];

                let counter_2_limit = if i == 0 { len - 1 } else { len };

                for j in (i + 2)..counter_2_limit {
                    let y1 = self.cities[j];
                    let y2 = self.cities[(j + 1) % len];

                    let expected_gain = Self::gain_from_2_opt(x1, x2, y1, y2, distances);
                    if expected_gain > best_move_gain {
                        best_move_gain = expected_gain;
                        best_move = Some(TwoOptMove {
                            i: TourIndex::new(i),
                            j: TourIndex::new(j),
                        });
                    }
                }
            }

            // If there is any move that shortens the tour, make it.
            if let Some(move2) = best_move {
                self.make_2_opt_move(move2.i, move2.j, best_move_gain);
            } else {
                // There are no moves that shorten the tour.
                break;
            }
        }
    }

    pub fn is_shorter_than(&self, other: &Tour) -> bool {
        self.tour_length < other.tour_length
    }

    pub fn get_path(&self, i1: TourIndex, i2: TourIndex) -> (CityIndex, CityIndex) {
        let (i1, i2) = (i1.get(), i2.get());
        assert!(i2 - i1 == 1 || i2 - i1 == self.cities.len() - 1);

        (self.cities[i1], self.cities[i2])
    }

    // Returns iterator over all paths except for the
    // first -> last.
    pub fn paths(&self) -> Windows<'_, CityIndex> {
        self.cities.windows(2)
    }

    pub fn cities(&self) -> &[CityIndex] {
        &self.cities
    }

    pub fn save_to_file<P: AsRef<Path>, Dp: AsRef<Path> + Display>(
        &self,
        problem_path: Dp,
        path: P,
    ) {
        let file = File::create(path).unwrap();
        let mut file = BufWriter::new(file);

        writeln!(file, "Problem file: {problem_path}").unwrap();
        writeln!(file, "Number of cities: {}", self.city_count()).unwrap();
        writeln!(file, "Tour length: {}", self.tour_length).unwrap();
        writeln!(file, "Cities:").unwrap();

        // Use indices starting at 1 for output, same as TSPLIB
        // format for consistency.
        for city in &self.cities {
            write!(file, "{} ", city.get() + 1).unwrap();
        }
        writeln!(file).unwrap();
    }
}

pub trait Length {
    fn calculate_tour_length(&self, distances: &SquareMatrix<f64>) -> f64;
}

impl Length for [CityIndex] {
    fn calculate_tour_length(&self, distances: &SquareMatrix<f64>) -> f64 {
        assert!(self.len() > 1);

        let mut tour_length = 0.0;
        for idx in 0..(self.len() - 1) {
            tour_length += distances[(self[idx].get(), self[idx + 1].get())];
        }
        // Add distance from last to first.
        tour_length += distances[(self.last().unwrap().get(), self[0].get())];

        tour_length
    }
}
