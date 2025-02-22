use activations::ActivationTrait;
use matrix::Matrix;

use crate::optimizr;

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
    pub m_weights: Matrix,  // First moment estimate
    pub v_weights: Matrix,  // Second moment estimate
    pub m_biases: Matrix,   // First moment estimate for biases
    pub v_biases: Matrix,   // Second moment estimate for biases
    pub optimizer_step: usize,
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

    pub fn update_weights(
        &mut self,
        d_weights: &Matrix,
        d_biases: &Matrix,
        learning_rate: f64,
        optimizer: optimizr::Optimizer,
    ) {
        match optimizer {
            optimizr::Optimizer::Adam => {
                let beta1 = 0.9;
                let beta2 = 0.999;
                let epsilon = 1e-8;

                self.optimizer_step += 1;
                let t = self.optimizer_step as f64;

                // Update biased first moment estimates
                self.m_weights = &(&self.m_weights * beta1) + &(d_weights * (1.0 - beta1));
                self.m_biases = &(&self.m_biases * beta1) + &(d_biases * (1.0 - beta1));

                // Update biased second moment estimates
                self.v_weights = &(&self.v_weights * beta2) + &(d_weights.hadamard(d_weights) * (1.0 - beta2));
                self.v_biases = &(&self.v_biases * beta2) + &(d_biases.hadamard(d_biases) * (1.0 - beta2));

                // Bias correction
                let beta1_correction = 1.0 - beta1.powf(t);
                let beta2_correction = 1.0 - beta2.powf(t);

                let m_hat_weights = &self.m_weights / beta1_correction;
                let v_hat_weights = &self.v_weights / beta2_correction;
                let m_hat_biases = &self.m_biases / beta1_correction;
                let v_hat_biases = &self.v_biases / beta2_correction;

                // Update weights and biases
                self.weights -= m_hat_weights / ((v_hat_weights.sqrt() + epsilon) * learning_rate);
                self.biases -= m_hat_biases / ((v_hat_biases.sqrt() + epsilon) * learning_rate);
            },
            optimizr::Optimizer::RMSprop => {
                let decay_rate = 0.99;
                let epsilon = 1e-8;

                // Update squared gradient estimate
                self.v_weights = &(&self.v_weights * decay_rate) + &(d_weights.hadamard(d_weights) * (1.0 - decay_rate));
                self.v_biases = &(&self.v_biases * decay_rate) + &(d_biases.hadamard(d_biases) * (1.0 - decay_rate));

                self.weights -= &(d_weights / ((self.v_weights.sqrt() + epsilon) * learning_rate));
                self.biases -= &(d_biases / ((self.v_biases.sqrt() + epsilon) * learning_rate));
            },
            optimizr::Optimizer::SGD => {
                // Simple gradient descent update
                self.weights = &self.weights - &(d_weights * learning_rate);
                self.biases = &self.biases - &(d_biases * learning_rate);
            }
        }
    }

    // construct the dense layer
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
            layer_type: LayerType::Dense, 
            m_weights: Matrix::zeros(0, 0),
            v_weights: Matrix::zeros(0, 0),
            m_biases: Matrix::zeros(0, 0),
            v_biases: Matrix::zeros(0, 0),
            optimizer_step:0,
        }
    }

    pub fn init_weights_and_biases(&mut self, input_dim: usize, output_dim: usize) {
        self.weights = Matrix::random(input_dim, output_dim);
        self.biases = Matrix::zeros(1, output_dim);
        self.m_weights = Matrix::zeros(input_dim, output_dim);
        self.v_weights = Matrix::zeros(input_dim, output_dim);
        self.m_biases = Matrix::zeros(1, output_dim);
        self.v_biases = Matrix::zeros(1, output_dim);
    }

    // construct the convolutional layer
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
            m_weights: Matrix::zeros(weight_rows, weight_cols),
            v_weights: Matrix::zeros(weight_rows, weight_cols),
            m_biases: Matrix::zeros(1, num_filters),
            v_biases: Matrix::zeros(1, num_filters),
            optimizer_step: 0,
        }
    }

}