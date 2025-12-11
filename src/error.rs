use std::fmt;

#[derive(Debug)]
pub struct DimensionError {
    dim_one: [usize; 2],
    dim_two: [usize; 2],
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix of {} x {} is incompatable with matrix of {} x {}.",
            self.dim_one[0], self.dim_one[1], self.dim_two[0], self.dim_two[1]
        )
    }
}

impl DimensionError {
    pub fn new(dim_one: [usize; 2], dim_two: [usize; 2]) -> Self {
        Self { dim_one, dim_two }
    }
}
