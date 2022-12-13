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

    pub fn wrapping_add(self, value: usize, len: usize) -> TourIndex {
        TourIndex::new((self.0 + value) % len)
    }
}

#[derive(Clone, Debug)]
pub struct Tour {
    cities: Vec<CityIndex>,
    // Don't look bits, used for 2-opt nad 3-opt.
    // Indexed by CityIndex.
    dont_look_bits: Vec<bool>,
    tour_length: u32,
}

impl Tour {
    pub const PLACEHOLDER: Tour = Tour {
        cities: Vec::new(),
        dont_look_bits: Vec::new(),
        tour_length: u32::MAX,
    };

    pub fn number_of_cities(&self) -> usize {
        self.cities.len()
    }

    pub fn length(&self) -> u32 {
        self.tour_length
    }

    pub fn from_cities(cities: Vec<CityIndex>, distances: &SquareMatrix<u32>) -> Tour {
        let tour_length = cities.calculate_tour_length(distances);
        let dont_look_bits = vec![false; cities.len()];

        Tour {
            cities,
            dont_look_bits,
            tour_length,
        }
    }

    pub fn from_hack_cities(mut cities_with_length: Vec<CityIndex>) -> Tour {
        let tour_length_usize = cities_with_length.pop().unwrap().get();
        let tour_length = tour_length_usize as u32;
        // Cities don't contain length anymore.
        let cities = cities_with_length;
        let dont_look_bits = vec![false; cities.len()];

        Tour {
            cities,
            dont_look_bits,
            tour_length,
        }
    }

    pub fn random(city_count: usize, distances: &SquareMatrix<u32>, rng: &mut impl Rng) -> Tour {
        assert!(city_count > 1);

        let mut cities: Vec<CityIndex> = (0..city_count).map(CityIndex::new).collect();
        cities.shuffle(rng);
        let tour_length = cities.calculate_tour_length(distances);
        let dont_look_bits = vec![false; cities.len()];

        Tour {
            cities,
            dont_look_bits,
            tour_length,
        }
    }

    pub fn from_prob_matrix(
        city_count: usize,
        probability_matrix: &ProbabilityMatrix,
        distances: &SquareMatrix<u32>,
        rng: &mut impl Rng,
    ) -> Tour {
        let mut cities = Vec::with_capacity(city_count);
        // Tracks whether each city is used. True = unused, false = used.
        let mut available_cities = vec![true; city_count];

        let starting_city = rng.gen_range(0..city_count);
        cities.push(CityIndex::new(starting_city));
        available_cities[starting_city] = false;
        // Keep track of the last city added to the tour.
        let mut last_added_city = starting_city;

        // How many cities are still missing.
        let mut cities_left = city_count - 1;

        'outermost: while cities_left > 0 {
            // Allow trying to insert the city `cities_left` times, then,
            // if still unsuccessful, insert the city with highest probability.
            for _ in 0..(cities_left * 2) {
                // Generate indices in unused cities only to avoid duplicates.
                let index = rng.gen_range(1..=cities_left);
                let prob = rng.gen::<f64>();
                // Check for unused cities and choose index-th unused city.
                // optimized
                // Find n-th 'true' in available_cities and try to insert it.
                let mut available: usize = 0;
                for (idx, unused) in available_cities.iter_mut().enumerate() {
                    available += (*unused) as usize;
                    if available == index {
                        let (l, h) = order(last_added_city, idx);
                        if prob <= probability_matrix[(h, l)] {
                            let city = CityIndex::new(idx);

                            cities.push(city);
                            last_added_city = idx;
                            // This causes false positive warning #[warn(clippy::mut_range_bound)]
                            // It is false positive, because we don't intend
                            // to affect the loop count of the outer `for`.
                            cities_left -= 1;
                            // Mark this city as used.
                            *unused = false;
                            continue 'outermost;
                        } else {
                            break;
                        }
                    }
                }
            }

            // If the control flow reaches here, insert city with highest prob.
            let (mut max_prob, mut max_prob_city, mut max_prob_dist) = (0.0, 0, u32::MAX);

            for (idx, unused) in available_cities.iter_mut().enumerate() {
                if *unused {
                    let (l, h) = order(last_added_city, idx);
                    let prob = probability_matrix[(h, l)];
                    if prob > max_prob {
                        let dist = distances[(h, l)];
                        (max_prob, max_prob_city, max_prob_dist) = (prob, idx, dist);
                    } else if prob == max_prob {
                        // If probabilities are the same, insert nearer city.
                        let dist = distances[(h, l)];
                        if dist < max_prob_dist {
                            (max_prob, max_prob_city, max_prob_dist) = (prob, idx, dist);
                        }
                    }
                }
            }
            cities.push(CityIndex::new(max_prob_city));
            // Mark the city as used.
            available_cities[max_prob_city] = false;
            cities_left -= 1;
        }

        // assert_eq!(cities.len(), city_count);
        // Check that there are no duplicates.
        // println!("{:?}", cities);
        // assert!(!(1..cities.len()).any(|i| cities[i..].contains(&cities[i - 1])));

        let tour_length = cities.calculate_tour_length(distances);
        let dont_look_bits = vec![false; city_count];

        Tour {
            cities,
            dont_look_bits,
            tour_length,
        }
    }

    // This function call must be matched by the corresponding
    // call to remove_hack_length().
    pub fn hack_append_length_at_tour_end(&mut self) {
        // This is only valid on >=32 bit architectures.
        let length_usize = self.tour_length as usize;
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
        distances: &SquareMatrix<u32>,
    ) -> i32 {
        let (x1, x2, y1, y2) = (x1.get(), x2.get(), y1.get(), y2.get());

        let del_length = distances[(x1, x2)] + distances[(y1, y2)];
        let add_length = distances[(x1, y1)] + distances[(x2, y2)];

        // Gain can be < 0, use i32.
        del_length as i32 - add_length as i32
    }

    fn make_2_opt_move(&mut self, i: TourIndex, j: TourIndex, move_gain: u32) {
        self.reverse_segment(self.successor(i), j);
        self.tour_length -= move_gain;
    }

    // Make 2-opt moves until no improvements can be made.
    // Choose the first move that gives any benefit.
    pub fn two_opt_dlb(&mut self, distances: &SquareMatrix<u32>) {
        let len = self.cities.len();
        let mut locally_optimal = false;

        while !locally_optimal {
            locally_optimal = true;
            'outermost_for: for city1_pos in 0..len {
                let first_city = self.cities[city1_pos];
                // If the DLB bit is set for the current city, we won't
                // find any unique improving moves from this city
                // Improving moves involving it may still be found from
                // other cities.
                if self.is_dlb_set(first_city) {
                    continue;
                }

                // We need to go in both forward and backward directions.
                // Alternative: set DLB to off if the segment with the city is reversed.

                // We either start from the first chosen city or its predecessor.
                // Reason: segment containing both first_city and its neighbour
                // could have been reversed and if we don't consider first_city's
                // predecessor (which could have been its successor previously),
                // we could miss some moves by ignoring first_city if it has DLB set.
                // 0 = forward, 1 = backward.
                for start_city_pos in [city1_pos, (city1_pos + len - 1) % len] {
                    let (x1, x2) = self.get_subsequent_pair(TourIndex::new(start_city_pos));

                    for city2_pos in 0..len {
                        let (y1, y2) = self.get_subsequent_pair(TourIndex::new(city2_pos));

                        // Since we are iterating over the same array as the
                        // loop above, we will get the same elements in one iteration and only one element ahead in another.
                        if x1 == y1 || x2 == y1 || y2 == x1 {
                            continue;
                        }

                        let expected_gain = Self::gain_from_2_opt(x1, x2, y1, y2, distances);
                        if expected_gain > 0 {
                            // If the move is beneficial, clear DLB for the
                            // cities involved and make the move.
                            self.clear_dlb(x1);
                            self.clear_dlb(x2);
                            self.clear_dlb(y1);
                            self.clear_dlb(y2);

                            self.make_2_opt_move(
                                TourIndex::new(start_city_pos),
                                TourIndex::new(city2_pos),
                                expected_gain as u32,
                            );
                            locally_optimal = false;
                            // Search the next first city for improvements.
                            continue 'outermost_for;
                        }
                    }
                }
                // If we reach here, we didn't find a valid move,
                // so set the DLB for the city searched.
                self.set_dlb(first_city);
            }
        }
    }

    // Is DLB set for this city?
    fn is_dlb_set(&self, city: CityIndex) -> bool {
        self.dont_look_bits[city.get()]
    }

    // Set DLB to false for this city.
    fn clear_dlb(&mut self, city: CityIndex) {
        self.dont_look_bits[city.get()] = false;
    }

    // Set DLB to true for this city.
    fn set_dlb(&mut self, city: CityIndex) {
        self.dont_look_bits[city.get()] = true;
    }

    // Make 2-opt moves until no improvements can be made.
    // Choose the first move that gives any benefit.
    pub fn two_opt(&mut self, distances: &SquareMatrix<u32>) {
        let len = self.cities.len();
        let mut locally_optimal = false;

        while !locally_optimal {
            locally_optimal = true;
            for i in 0..(len - 2) {
                let (x1, x2) = self.get_subsequent_pair(TourIndex::new(i));

                let counter_2_limit = if i == 0 { len - 1 } else { len };

                for j in (i + 2)..counter_2_limit {
                    let (y1, y2) = self.get_subsequent_pair(TourIndex::new(j));

                    let expected_gain = Self::gain_from_2_opt(x1, x2, y1, y2, distances);
                    if expected_gain > 0 {
                        // If the move is beneficial, make it.
                        self.make_2_opt_move(
                            TourIndex::new(i),
                            TourIndex::new(j),
                            expected_gain as u32,
                        );
                        locally_optimal = false;
                        break;
                    }
                }
            }
        }
    }

    // Make 2-opt moves until no improvements can be made.
    // Choose the best possible move each time.
    pub fn two_opt_take_best_each_time(&mut self, distances: &SquareMatrix<u32>) {
        #[derive(Debug, Clone, Copy)]
        struct TwoOptMove {
            i: TourIndex,
            j: TourIndex,
        }

        let len = self.cities.len();

        loop {
            let mut best_move_gain = 0;
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
                self.make_2_opt_move(move2.i, move2.j, best_move_gain as u32);
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
        distances: &SquareMatrix<u32>,
    ) -> i32 {
        let (x1, x2, y1, y2, z1, z2) = (x1.get(), x2.get(), y1.get(), y2.get(), z1.get(), z2.get());

        let (del_length, add_length) = match reconnection_case {
            ThreeOptReconnectionCase::A_B_C => return 0,
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

        // Gain can be < 0, so use i32.
        del_length as i32 - add_length as i32
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
        move_gain: u32,
        // distances: &SquareMatrix<u32>,
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

        // if u32::abs(tour_len_before - move_gain - tour_len_after) > 0.0001 {
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

    pub fn three_opt(&mut self, distances: &SquareMatrix<u32>) {
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

                            if expected_gain > 0 {
                                self.make_3_opt_move(
                                    TourIndex::new(i),
                                    TourIndex::new(j),
                                    TourIndex::new(k),
                                    c,
                                    // expected_gain > 0
                                    expected_gain as u32,
                                    /*distances,*/
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

    pub fn three_opt_dlb(&mut self, distances: &SquareMatrix<u32>) {
        let mut locally_optimal = false;
        let len = self.number_of_cities();

        dbg!();

        'outermost: while !locally_optimal {
            locally_optimal = true;
            for preliminary_city_1_pos in (0..len).map(TourIndex::new) {
                let first_city = self.cities[preliminary_city_1_pos.get()];
                if self.is_dlb_set(first_city) {
                    continue;
                }

                for city_1_pos in [
                    preliminary_city_1_pos,
                    preliminary_city_1_pos.wrapping_add(len - 1, len),
                ] {
                    let (x1, x2) = self.get_subsequent_pair(city_1_pos);

                    // If either of the first two cities has DLB set, skip them.
                    if self.is_dlb_set(x1) || self.is_dlb_set(x2) {
                        continue;
                    }

                    for city_2_pos in (0..len).map(TourIndex::new) {
                        let (y1, y2) = self.get_subsequent_pair(city_2_pos);

                        // Don't select the same or subsequent cities as outer loop.
                        if x1 == y1 || x2 == y1 || y2 == x1 {
                            continue;
                        }

                        // Examine 2-opt case, we only need two cities for it.
                        let expected_gain = Self::gain_from_2_opt(x1, x2, y1, y2, distances);
                        if expected_gain > 0 {
                            self.make_3_opt_move(
                                preliminary_city_1_pos,
                                city_2_pos,
                                city_2_pos,
                                ThreeOptReconnectionCase::RevA_B_C,
                                expected_gain as u32,
                            );

                            // Improvement has been found, look for next.
                            locally_optimal = false;
                            // Clear DLB for involved cities.
                            self.clear_dlb(x1);
                            self.clear_dlb(x2);
                            self.clear_dlb(y1);
                            self.clear_dlb(y2);

                            continue 'outermost;
                        }

                        for city_3_pos in (0..len).map(TourIndex::new) {
                            let (z1, z2) = self.get_subsequent_pair(city_3_pos);

                            // Don't select the same cities as outer loops.
                            if x1 == z1 || y1 == z1 {
                                continue;
                            }

                            // Second city must be between first and third.
                            if !Self::between(city_1_pos, city_2_pos, city_3_pos) {
                                continue;
                            }

                            for reconnection_case in [
                                ThreeOptReconnectionCase::RevA_B_RevC,
                                ThreeOptReconnectionCase::RevA_RevB_RevC,
                            ] {
                                let expected_gain = Self::gain_from_3_opt(
                                    x1,
                                    x2,
                                    y1,
                                    y2,
                                    z1,
                                    z2,
                                    reconnection_case,
                                    distances,
                                );

                                if expected_gain > 0 {
                                    self.make_3_opt_move(
                                        city_1_pos,
                                        city_2_pos,
                                        city_3_pos,
                                        reconnection_case,
                                        expected_gain as u32,
                                    );

                                    // Improvement has been found, look for next.
                                    locally_optimal = false;
                                    // Clear DLB for involved cities.
                                    self.clear_dlb(x1);
                                    self.clear_dlb(x2);
                                    self.clear_dlb(y1);
                                    self.clear_dlb(y2);
                                    self.clear_dlb(z1);
                                    self.clear_dlb(z2);

                                    continue 'outermost;
                                }
                            }
                        }
                    }
                }
                // No moves have been found for this city, set DLB for it.
                self.set_dlb(first_city);
            }
        }
    }

    // Returns true if x is between a and b in the tour:
    // we start at position a in the tour (in forward direction),
    // and x is between a and b if we encounter it sooner than b.
    fn between(a: TourIndex, x: TourIndex, b: TourIndex) -> bool {
        if b > a {
            // x is simply between a and b
            (x > a) && (x < b)
        } else if b < a {
            // b is before a in the tour, so we need to take into account
            // that x can be more than a or less than b
            (x > a) || (x < b)
        } else {
            // b == a
            false
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
    // Noop.
    A_B_C = 0,
    // Equivalent to 2-opt move with x1, x2 the same, y1 = z1, y2 = z2.
    RevA_B_C = 1,
    // Equivalent to 2-opt move with y1, y2 the same, x1 = z1, x2 = z2.
    A_B_RevC = 2,
    // Equivalent to 2-opt move with x1, x2, y1, y2 the same.
    A_RevB_C = 3,
    A_RevB_RevC = 4,
    RevA_RevB_C = 5,
    RevA_B_RevC = 6,
    RevA_RevB_RevC = 7,
}

pub trait Length {
    fn calculate_tour_length(&self, distances: &SquareMatrix<u32>) -> u32;

    fn hack_get_tour_length_from_last_element(&self) -> u32;
}

impl Length for [CityIndex] {
    fn calculate_tour_length(&self, distances: &SquareMatrix<u32>) -> u32 {
        assert!(self.len() > 1);

        let mut tour_length = 0;
        for idx in 1..self.len() {
            tour_length += distances[(self[idx - 1].get(), self[idx].get())];
        }
        // Add distance from last to first.
        tour_length += distances[(self.last().unwrap().get(), self[0].get())];

        tour_length
    }

    // The length must first be inserted using Tour::hack_append_length_at_tour_end().
    fn hack_get_tour_length_from_last_element(&self) -> u32 {
        let tour_length_usize = self.last().unwrap().get();

        tour_length_usize as u32
    }
}
