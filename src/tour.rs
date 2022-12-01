use std::fmt::Display;
use std::io::Write;
use std::{fs::File, io::BufWriter, path::Path, slice::Windows};

use rand::seq::SliceRandom;
use rand::Rng;

use crate::matrix::SquareMatrix;
use crate::order;
use crate::probability_matrix::ProbabilityMatrix;
use crate::tsp_solver::CityIndex;

// Position of city in the tour. Zero-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TourIndex(usize);

impl TourIndex {
    pub fn new(index: usize) -> TourIndex {
        TourIndex(index)
    }

    pub fn get(&self) -> usize {
        self.0
    }
}

#[cfg(not(target_pointer_width = "64"))]
compile_error!("Hack with tour length only works on >=64 bit architectures");

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

    pub fn number_of_cities(&self) -> usize {
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

    pub fn from_hack_cities(mut cities_with_length: Vec<CityIndex>) -> Tour {
        let tour_length_usize = cities_with_length.pop().unwrap().get();
        let tour_length = f64::from_bits(tour_length_usize as u64);

        Tour {
            cities: cities_with_length,
            tour_length,
        }
    }

    pub fn random(city_count: usize, distances: &SquareMatrix<f64>, rng: &mut impl Rng) -> Tour {
        assert!(city_count > 1);

        let mut cities: Vec<CityIndex> = (0..city_count).map(CityIndex::new).collect();
        cities.shuffle(rng);
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    pub fn from_prob_matrix(
        city_count: usize,
        probability_matrix: &ProbabilityMatrix,
        distances: &SquareMatrix<f64>,
        rng: &mut impl Rng,
    ) -> Tour {
        let mut cities = Vec::with_capacity(city_count);

        let starting_city = rng.gen_range(0..city_count);
        cities.push(CityIndex::new(starting_city));
        // How many cities are still missing.
        let mut cities_left = city_count - 1;

        // println!("reached line {} in {}", line!(), file!());

        'outermost: while cities_left > 0 {
            let last: usize = cities.last().unwrap().get();

            // Allow trying to insert the city `cities_left` times, then,
            // if still unsuccessful, insert the city with highest probability.
            for _ in 0..cities_left {
                // Generate indices in unused cities only to avoid duplicates.
                let index = rng.gen_range(0..cities_left);
                let prob = rng.gen::<f64>();

                // Check for unused cities and choose index-th unused city.
                let mut seen_unused_city_count = 0;
                for c in 0..city_count {
                    let city_index = CityIndex::new(c);
                    if !cities.contains(&city_index) {
                        if seen_unused_city_count == index {
                            let (l, h) = order(last, c);
                            if prob <= probability_matrix[(h, l)] {
                                cities.push(city_index);
                                // This causes false positive warning #[warn(clippy::mut_range_bound)]
                                // It is false positive, because we don't intend
                                // to affect the loop count of this `for`.
                                cities_left -= 1;
                                continue 'outermost;
                            }

                            // Try to insert another city.
                            break;
                        }
                        seen_unused_city_count += 1;
                    }
                }
            }
            // If the control flow reaches here, insert city with highest prob.
            let (mut max_prob, mut max_prob_city, mut max_prob_dist) = (0.0, 0, f64::INFINITY);
            for _ in 0..cities_left {
                for c in 0..city_count {
                    let city_index = CityIndex::new(c);
                    if !cities.contains(&city_index) {
                        let (l, h) = order(last, c);
                        let prob = probability_matrix[(h, l)];
                        if prob > max_prob {
                            let dist = distances[(h, l)];
                            (max_prob, max_prob_city, max_prob_dist) = (prob, c, dist);
                        } else if prob == max_prob {
                            // If probabilities are the same, insert nearer city.
                            let dist = distances[(h, l)];
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

        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    // This function call must be matched by the corresponding
    // call to remove_hack_length().
    pub fn hack_append_length_at_tour_end(&mut self) {
        let length_u64 = self.tour_length.to_bits();
        // println!("Tour length u64: {length_u64}");
        // This is only valid on 64 bit architectures
        let length_usize = length_u64 as usize;
        let length_index = CityIndex::new(length_usize);
        self.cities.push(length_index);
    }

    pub fn remove_hack_length(&mut self) {
        self.cities.pop();
    }

    fn reverse_segment(&mut self, start: TourIndex, end: TourIndex) {
        let mut left = start.get();
        let mut right = end.get();
        let len = self.cities.len();

        let inversion_size = ((len + right - left + 1) % len) / 2;
        // dbg!(inversion_size);
        // assert!(inversion_size >= 1);
        // if inversion_size == 0 {
        //     dbg!();
        // }

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

    fn make_2_opt_move(&mut self, i: TourIndex, j: TourIndex, move_gain: f64) {
        self.reverse_segment(self.successor(i), j);
        // This is not perfectly accurate due to float rounding,
        // but will be good enough.
        self.tour_length -= move_gain;
    }

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
                let (x1, x2) = self.get_subsequent_pair(TourIndex::new(i));

                let counter_2_limit = if i == 0 { len - 1 } else { len };

                for j in (i + 2)..counter_2_limit {
                    let (y1, y2) = self.get_subsequent_pair(TourIndex::new(j));

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

    pub fn last_to_first_path(&self) -> (CityIndex, CityIndex) {
        (*self.cities.last().unwrap(), self.cities[0])
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
        writeln!(file, "Number of cities: {}", self.number_of_cities()).unwrap();
        writeln!(file, "Tour length: {}", self.tour_length).unwrap();
        writeln!(file, "Cities:").unwrap();

        // Use indices starting at 1 for output, same as TSPLIB
        // format for consistency.
        for city in &self.cities {
            write!(file, "{} ", city.get() + 1).unwrap();
        }
        writeln!(file).unwrap();
    }

    // The bigger, the better.
    fn gain_from_3_opt(
        x1: CityIndex,
        x2: CityIndex,
        y1: CityIndex,
        y2: CityIndex,
        z1: CityIndex,
        z2: CityIndex,
        reconnection_case: ThreeOptReconnectionCase,
        distances: &SquareMatrix<f64>,
    ) -> f64 {
        let (x1, x2, y1, y2, z1, z2) = (x1.get(), x2.get(), y1.get(), y2.get(), z1.get(), z2.get());

        let (del_length, add_length) = match reconnection_case {
            ThreeOptReconnectionCase::A_B_C => return 0.0,
            ThreeOptReconnectionCase::RevA_B_C => (
                distances[(x1, x2)] + distances[(z1, z2)],
                distances[(x1, z1)] + distances[(x2, z2)],
            ),
            ThreeOptReconnectionCase::A_B_RevC => (
                distances[(y1, y2)] + distances[(z1, z2)],
                distances[(y1, z1)] + distances[(y2, z2)],
            ),
            ThreeOptReconnectionCase::A_RevB_C => (
                distances[(x1, x2)] + distances[(y1, y2)],
                distances[(x1, y1)] + distances[(x2, y2)],
            ),
            ThreeOptReconnectionCase::A_RevB_RevC => (
                distances[(x1, x2)] + distances[(y1, y2)] + distances[(z1, z2)],
                distances[(x1, y1)] + distances[(x2, z1)] + distances[(y2, z2)],
            ),
            ThreeOptReconnectionCase::RevA_RevB_C => (
                distances[(x1, x2)] + distances[(y1, y2)] + distances[(z1, z2)],
                distances[(x1, z1)] + distances[(x2, y2)] + distances[(y1, z2)],
            ),
            ThreeOptReconnectionCase::RevA_B_RevC => (
                distances[(x1, x2)] + distances[(y1, y2)] + distances[(z1, z2)],
                distances[(x1, y2)] + distances[(x2, z2)] + distances[(y1, z1)],
            ),
            ThreeOptReconnectionCase::RevA_RevB_RevC => (
                distances[(x1, x2)] + distances[(y1, y2)] + distances[(z1, z2)],
                distances[(x1, y2)] + distances[(x2, z1)] + distances[(y1, z2)],
            ),
        };

        del_length - add_length
    }

    // x1 = i, x2 = successor(i)
    // y1 = j, y2 = successor(j)
    // z1 = k, z2 = successor(k)
    // Connections:
    // z2 - a - x1
    // x2 - b - y1
    // y2 - c - z1
    fn make_3_opt_move(
        &mut self,
        i: TourIndex,
        j: TourIndex,
        k: TourIndex,
        reconnection_case: ThreeOptReconnectionCase,
        move_gain: f64,
        distances: &SquareMatrix<f64>,
    ) {
        // let tour_len_before = self.cities.calculate_tour_length(distances);
        match reconnection_case {
            ThreeOptReconnectionCase::A_B_C => (),
            ThreeOptReconnectionCase::RevA_B_C => self.reverse_segment(self.successor(k), i),
            ThreeOptReconnectionCase::A_B_RevC => self.reverse_segment(self.successor(j), k),
            ThreeOptReconnectionCase::A_RevB_C => self.reverse_segment(self.successor(i), j),
            ThreeOptReconnectionCase::A_RevB_RevC => {
                self.reverse_segment(self.successor(j), k);
                self.reverse_segment(self.successor(i), j);
            }
            ThreeOptReconnectionCase::RevA_RevB_C => {
                self.reverse_segment(self.successor(k), i);
                self.reverse_segment(self.successor(i), j);
            }
            ThreeOptReconnectionCase::RevA_B_RevC => {
                self.reverse_segment(self.successor(k), i);
                self.reverse_segment(self.successor(j), k);
            }
            ThreeOptReconnectionCase::RevA_RevB_RevC => {
                self.reverse_segment(self.successor(k), i);
                self.reverse_segment(self.successor(i), j);
                self.reverse_segment(self.successor(j), k);
            }
        }

        // let tour_len_after = self.cities.calculate_tour_length(distances);

        // if f64::abs(tour_len_before - move_gain - tour_len_after) > 0.0001 {
        //     dbg!(i, j, k);
        //     dbg!(reconnection_case, tour_len_before, tour_len_after);
        // }
        self.tour_length -= move_gain;
    }

    fn successor(&self, i: TourIndex) -> TourIndex {
        TourIndex::new((i.get() + 1) % self.number_of_cities())
    }

    fn get_subsequent_pair(&self, i: TourIndex) -> (CityIndex, CityIndex) {
        let (i, j) = (i.get(), self.successor(i).get());

        (self.cities[i], self.cities[j])
    }

    pub fn three_opt(&mut self, distances: &SquareMatrix<f64>) {
        let mut locally_optimal = false;
        let len = self.number_of_cities();

        while !locally_optimal {
            locally_optimal = true;

            'for_i: for counter_1 in 0..len {
                let i = counter_1;
                let (x1, x2) = self.get_subsequent_pair(TourIndex::new(i));

                for counter_2 in 1..(len - 2) {
                    let j = (counter_2 + i) % len;
                    let (y1, y2) = self.get_subsequent_pair(TourIndex::new(j));

                    for counter_3 in (counter_2 + 1)..len {
                        let k = (counter_3 + i) % len;
                        assert_ne!(j, k);
                        let (z1, z2) = self.get_subsequent_pair(TourIndex::new(k));

                        for c in [
                            ThreeOptReconnectionCase::A_RevB_C,
                            ThreeOptReconnectionCase::RevA_B_RevC,
                            ThreeOptReconnectionCase::RevA_RevB_RevC,
                        ] {
                            let expected_gain =
                                Self::gain_from_3_opt(x1, x2, y1, y2, z1, z2, c, distances);

                            if expected_gain > 0.0000001 {
                                self.make_3_opt_move(
                                    TourIndex::new(i),
                                    TourIndex::new(j),
                                    TourIndex::new(k),
                                    c,
                                    expected_gain,
                                    distances,
                                );
                                // dbg!(expected_gain);
                                locally_optimal = false;
                                break 'for_i;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Segments a, b, c are connected by paths that we are replacing.
// Segments a, b, c are arranged in clockwise fashion:
//     x1 - x2
//    /       \
//   a         b
//  /           \
// z2           y1
//  \          /
//   z1 - c - y2
// Here x1, x2, y1, y2, z1, z2 - connection points.
// There are several possible ways to reconnect the disconnected segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ThreeOptReconnectionCase {
    A_B_C = 0,
    RevA_B_C = 1,
    A_B_RevC = 2,
    A_RevB_C = 3,
    A_RevB_RevC = 4,
    RevA_RevB_C = 5,
    RevA_B_RevC = 6,
    RevA_RevB_RevC = 7,
}

pub trait Length {
    fn calculate_tour_length(&self, distances: &SquareMatrix<f64>) -> f64;

    fn hack_get_tour_length_from_last_element(&self) -> f64;
}

impl Length for [CityIndex] {
    fn calculate_tour_length(&self, distances: &SquareMatrix<f64>) -> f64 {
        assert!(self.len() > 1);

        let mut tour_length = 0.0;
        for idx in 1..self.len() {
            tour_length += distances[(self[idx - 1].get(), self[idx].get())];
        }
        // Add distance from last to first.
        tour_length += distances[(self.last().unwrap().get(), self[0].get())];

        tour_length
    }

    // The length must first be inserted using Tour::hack_append_length_at_tour_end().
    fn hack_get_tour_length_from_last_element(&self) -> f64 {
        let tour_length_usize = self.last().unwrap().get();

        f64::from_bits(tour_length_usize as u64)
    }
}
