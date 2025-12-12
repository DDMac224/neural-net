use std::fmt;

use crate::{
    error::DimensionError,
    layer::{Layer, LayerError, Node},
};

#[derive(Debug)]
pub struct FFError {
    message: String,
}

impl From<DimensionError> for FFError {
    fn from(value: DimensionError) -> Self {
        Self {
            message: format!("{}", value),
        }
    }
}
impl From<LayerError> for FFError {
    fn from(value: LayerError) -> Self {
        Self {
            message: format!("{}", value),
        }
    }
}

impl fmt::Display for FFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn lin_forward<T: Node>(input: Layer<T>, weights: Vec<Layer<T>>) -> Result<Layer<T>, FFError> {
    let mut cache = input.clone();
    for layer in weights {
        cache = cache.mult(&layer)?;
        cache = cache.activate();
    }
    Ok(cache)
}
