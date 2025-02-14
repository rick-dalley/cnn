use activations::ActivationTrait;
use matrix::Matrix;

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
    pub layer_type: LayerType, // enum to handle specific properties
}

impl Layer {
    pub fn get_stride_size(&self) -> Option<usize> {
        match &self.layer_type {
            LayerType::Convolutional { stride_size, .. } => Some(*stride_size),
            _ => None, // dense layers don't use stride_size
        }
    }

    pub fn get_kernel_size(&self) -> Option<usize> {
        match &self.layer_type {
            LayerType::Convolutional { kernel_size, .. } => Some(*kernel_size),
            _ => None, // dense layers don't use kernel_size
        }
    }
    pub fn get_weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn get_biases(&self) -> &Matrix {
        &self.biases
    }

    pub fn set_weights(&mut self, weights: Matrix) {
        self.weights = weights;
    }

    pub fn set_biases(&mut self, biases: Matrix) {
        self.biases = biases;
    }

    // Constructor for Dense Layer
    pub fn new_dense(
        function_family: String,
        alpha: f64,
        lambda: f64,
    ) -> Self {
        let (activation_fn, derivative_fn) =
            activations::activations::get_activation_and_derivative(function_family, alpha, lambda);
            let weights = Matrix::new(0, 0, Vec::new());
            let biases = Matrix::new(0, 0, Vec::new());
        Self {
            weights,
            biases,
            activation_fn,
            derivative_fn,
            layer_type: LayerType::Dense, // No additional properties for Dense
        }
    }

    // Constructor for Convolutional Layer
    pub fn new_convolutional(
        num_filters: usize,  
        kernel_size: usize,
        stride_size: usize,
        function_family: String,
        alpha: f64,
        lambda: f64,
    ) -> Self {
        let (activation_fn, derivative_fn) =
            activations::activations::get_activation_and_derivative(function_family, alpha, lambda);

        let weight_rows = num_filters; // each row represents a filter
        let weight_cols = kernel_size * kernel_size; // the filter flattened

        Self {
            weights: Matrix::random(weight_rows, weight_cols),
            biases: Matrix::zeros(1, num_filters), // one bias per filter
            activation_fn,
            derivative_fn,
            layer_type: LayerType::Convolutional {
                stride_size,
                kernel_size,
            },
        }
    }

}