use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

// Table layout:
// [row, row, ..]
#[derive(Clone)]
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
    pub fn side_length(&self) -> usize {
        self.side_length
    }
}

impl<T> SquareMatrix<T>
where
    T: Copy,
{
    pub fn new(side_length: usize, init_value: T) -> SquareMatrix<T> {
        let data = vec![init_value; side_length * side_length];

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

impl Display for SquareMatrix<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--- SquareMatrix<f64> ---")?;
        for row in self.data.chunks_exact(self.side_length) {
            for elem in row {
                write!(f, "{elem:.2}\t")?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
