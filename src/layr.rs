use activations::activations::ActivationTrait;
use matrix::matrix::{Matrix, Dot};

#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
    Convolutional { stride_size: usize, kernel_size: usize },
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation_fn: Box<dyn ActivationTrait>,
    pub derivative_fn: Box<dyn ActivationTrait>,
    pub layer_type: LayerType, // Enum to handle specific properties
}

impl Layer {
    pub fn get_stride_size(&self) -> Option<usize> {
        match &self.layer_type {
            LayerType::Convolutional { stride_size, .. } => Some(*stride_size),
            _ => None, // Dense layers don't have stride_size
        }
    }

    pub fn get_kernel_size(&self) -> Option<usize> {
        match &self.layer_type {
            LayerType::Convolutional { kernel_size, .. } => Some(*kernel_size),
            _ => None, // Dense layers don't have kernel_size
        }
    }
    // Constructor for Dense Layer
    pub fn new_dense(
        input_dim: usize,
        output_dim: usize,
        function_family: String,
        alpha: f64,
        lambda: f64,
    ) -> Self {
        let (activation_fn, derivative_fn) =
            activations::activations::get_activation_and_derivative(function_family, alpha, lambda);
        Self {
            weights: Matrix::random(input_dim, output_dim),
            biases: Matrix::zeros(1, output_dim),
            activation_fn,
            derivative_fn,
            layer_type: LayerType::Dense, // No additional properties for Dense
        }
    }

    // Constructor for Convolutional Layer
    pub fn new_convolutional(
        num_filters: usize,   // This represents output channels
        kernel_size: usize,
        stride_size: usize,
        function_family: String,
        alpha: f64,
        lambda: f64,
    ) -> Self {
        let (activation_fn, derivative_fn) =
            activations::activations::get_activation_and_derivative(function_family, alpha, lambda);

        let weight_rows = num_filters; // Each row represents a filter
        let weight_cols = kernel_size * kernel_size; // Flattened filter

        Self {
            weights: Matrix::random(weight_rows, weight_cols),
            biases: Matrix::zeros(1, num_filters), // One bias per filter
            activation_fn,
            derivative_fn,
            layer_type: LayerType::Convolutional {
                stride_size,
                kernel_size,
            },
        }
    }

    pub fn apply_weights_and_biases(&self, patch: &Matrix) -> Matrix {
        if let LayerType::Convolutional { .. } = self.layer_type {
            let weighted = patch.dot(&self.weights.transpose()); // Apply weights
            let biased = &weighted + &self.biases; // Apply biases
            biased
        } else {
            panic!("apply_weights_and_biases called on non-convolutional layer!");
        }
    }

}