use std::str::FromStr;

use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    Adam,
    SGD,
    RMSprop,
}

impl FromStr for Optimizer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "adam" => Ok(Optimizer::Adam),
            "sgd" => Ok(Optimizer::SGD),
            "rmsprop" => Ok(Optimizer::RMSprop),
            _ => Err(format!("Invalid Optimizer: {}", s)),
        }
    }
}