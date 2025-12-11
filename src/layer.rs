use num_traits::NumOps;

use std::fmt::{self, Debug};

use crate::error::DimensionError;

#[derive(Debug)]
pub struct LayerError {
    message: String,
}

impl From<DimensionError> for LayerError {
    fn from(value: DimensionError) -> Self {
        Self {
            message: format!("{}", value),
        }
    }
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl LayerError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

pub trait Node: NumOps + Copy + PartialEq + Debug {}
impl<T: NumOps + Copy + PartialEq + Debug> Node for T {}

#[derive(Debug, Clone)]
pub struct Layer<T: Node> {
    data: Vec<T>,
    num_row: usize,
    num_col: usize,
    activation_func: fn(&T) -> T,
}

impl<T> PartialEq for Layer<T>
where
    T: Node,
{
    fn eq(&self, other: &Self) -> bool {
        *(other.data_vec()) == self.data
            && self.num_col == other.cols_count()
            && self.num_row == other.rows_count()
    }
}

impl<T> Layer<T>
where
    T: Node,
{
    pub fn new(
        data: Vec<T>,
        num_row: usize,
        num_col: usize,
        activation_func: Option<fn(&T) -> T>,
    ) -> Result<Self, LayerError> {
        if data.len() != num_col * num_row {
            return Err(LayerError::new(String::from(format!(
                "Incorrect data size {} != {} * {}",
                data.len(),
                num_col,
                num_row
            ))));
        }

        Ok(Self {
            data: data,
            num_row: num_row,
            num_col: num_col,
            activation_func: activation_func.unwrap_or(|&x| x),
        })
    }

    pub fn row(&self, idx: usize) -> impl Iterator<Item = &T> + '_ {
        self.data[self.num_col * idx..self.num_col * (idx + 1)].iter()
    }

    pub fn col(&self, idx: usize) -> impl Iterator<Item = &T> + '_ {
        self.data.iter().skip(idx).step_by(self.num_col)
    }

    pub fn rows_count(&self) -> usize {
        self.num_row
    }

    pub fn cols_count(&self) -> usize {
        self.num_col
    }

    pub fn data_vec(&self) -> &Vec<T> {
        &(self.data)
    }

    pub fn mult(&self, other: &Self) -> Result<Self, LayerError> {
        if self.num_col != other.rows_count() {
            return Err(DimensionError::new(
                [self.num_col, self.num_row],
                [other.cols_count(), other.rows_count()],
            )
            .into());
        }

        let mut output_data: Vec<T> = vec![];
        for row in 0..self.num_row {
            for col in 0..other.cols_count() {
                output_data.push(
                    self.row(row)
                        .zip(other.col(col))
                        .map(|(&x, &y)| x * y)
                        .reduce(|acc, e| acc + e)
                        .unwrap(),
                );
            }
        }

        Ok(Layer::new(
            output_data,
            self.num_row,
            other.cols_count(),
            Some(self.activation_func),
        )?)
    }

    pub fn activate(&self) -> Layer<T> {
        // Since data shape never changes unwrap should be safe
        Layer::new(
            self.data.iter().map(self.activation_func).collect(),
            self.num_row,
            self.num_col,
            None,
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::max;

    use super::*;

    #[test]
    fn multiply() {
        let m1 = Layer::new(vec![1, 2, 3, 4, 5, 6], 2, 3, None).unwrap();

        let m2 = Layer::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, None).unwrap();

        let result = m1.mult(&m2).unwrap();

        let expected = Layer::new(vec![30, 36, 42, 66, 81, 96], 2, 3, None).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn activate() {
        let m = Layer::new(
            vec![1, -2, 3, -4, 5, -6, 7, -8, 9],
            3,
            3,
            Some(|&x| max(0, x)),
        )
        .unwrap();
        let result = m.activate();
        let expected =
            Layer::new(vec![1, 0, 3, 0, 5, 0, 7, 0, 9], 3, 3, Some(|&x| max(0, x))).unwrap();

        assert_eq!(result, expected)
    }
}
