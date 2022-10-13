use std::{mem, slice::Windows};

use mpi::traits::Equivalence;
use rand::Rng;

use crate::{matrix::SquareMatrix, CityIndex};

// TODO: maybe use a type alias instead?
// right now this is really unergonomic

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
        let tour_length = cities.tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    // TODO: use rand::shuffle.
    pub fn random(city_count: usize, distances: &SquareMatrix<f64>, rng: &mut impl Rng) -> Tour {
        let mut cities = Vec::with_capacity(city_count);
        let mut tour_length = 0.0;
        let start = rng.gen_range(0..city_count);
        cities.push(CityIndex::new(start));

        for idx in 1..city_count {
            // Generate indices in unused cities only to avoid duplicates.
            let index = rng.gen_range(0..(city_count - idx));
            // Check for unused cities and choose index-th unused city.
            let mut unused_city_count = 0;
            for c in 0..city_count {
                let city_index = CityIndex(c);
                if !cities.contains(&city_index) {
                    if unused_city_count == index {
                        cities.push(city_index);
                        tour_length += distances[(cities[idx - 1].get(), cities[idx].get())];
                        break;
                    }
                    unused_city_count += 1;
                }
            }
        }

        Tour {
            cities,
            tour_length,
        }
    }
    /*
    pub fn nearest_neighbour(
        city_count: usize,
        starting_city: usize,
        distances: &SquareMatrix<f64>,
    ) -> Tour {
        assert!(city_count > starting_city);

        let mut cities = Vec::with_capacity(city_count);
        cities.push(starting_city);

        todo!()
    }*/

    // From https://tsp-basics.blogspot.com/2017/02/building-blocks-reversing-segment.html
    fn reverse_segment(&mut self, start: TourIndex, end: TourIndex) {
        let mut left = start.get();
        let mut right = end.get();
        let len = self.cities.len();

        let inversion_size = ((len + end.get() - start.get() + 1) % len) / 2;
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
        let del_length = distances[(x1.get(), x2.get())] + distances[(y1.get(), y2.get())];
        let add_length = distances[(x1.get(), y1.get())] + distances[(x2.get(), y2.get())];

        del_length - add_length
    }

    // From https://tsp-basics.blogspot.com/2017/03/2-opt-move.html
    // a..b in Nim is inclusive!!!
    // https://nim-lang.org/docs/tut1.html#control-flow-statements-for-statement
    fn make_2_opt_move(&mut self, mut i: TourIndex, j: TourIndex) {
        self.reverse_segment(i.inc(self.cities.len()), j);
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

    pub fn ls_2_opt_take_best(&mut self, distances: &SquareMatrix<f64>) {
        #[derive(Debug, Clone, Copy)]
        struct TwoOptMove {
            i: TourIndex,
            j: TourIndex,
        }

        let len = self.cities.len();

        loop {
            // println!("reached line {} in {}", line!(), file!());

            let mut best_move_gain = 0.0;
            // There might not be any moves that shorten the tour.
            let mut best_move: Option<TwoOptMove> = None;

            for counter_1 in 0..(len - 2) {
                let i = counter_1;
                let x1 = self.cities[i];
                let x2 = self.cities[(i + 1) % len];

                let counter_2_limit = if i == 0 { len - 1 } else { len };

                for counter_2 in (i + 2)..counter_2_limit {
                    let j = counter_2;
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
                self.make_2_opt_move(move2.i, move2.j);
            } else {
                // There are no moves that shorten the tour.
                break;
            }
        }
    }

    /*
    pub fn has_path(&self, ci1: CityIndex, ci2: CityIndex) -> bool {
        self.cities.windows(2).any(|indices| {
            if let [i1, i2] = *indices {
                (i1 == ci1 && ci2 == i2) || (i1 == ci2 && i2 == ci1)
            } else {
                unreachable!()
            }
        })
    }*/

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
}

pub trait Length {
    fn tour_length(&self, distances: &SquareMatrix<f64>) -> f64;
}

impl Length for [CityIndex] {
    fn tour_length(&self, distances: &SquareMatrix<f64>) -> f64 {
        assert!(!self.len() > 1);

        let mut tour_length = 0.0;
        for idx in 0..(self.len() - 1) {
            tour_length += distances[(self[idx].get(), self[idx + 1].get())];
        }
        // Add distance from last to first.
        tour_length += distances[(self.last().unwrap().get(), self[0].get())];

        tour_length
    }
}
