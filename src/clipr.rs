use std::collections::VecDeque;
use matrix::Matrix; // Ensure this is the correct path for your Matrix module

#[derive(Debug, Clone)]
pub struct AdaptiveGradientClipping {
    window_size: usize,
    factor: f64,
    grad_norms: VecDeque<f64>,
}

impl AdaptiveGradientClipping {
    pub fn new(window_size: usize, factor: f64) -> Self {
        Self {
            window_size,
            factor,
            grad_norms: VecDeque::with_capacity(window_size),
        }
    }

    pub fn update_and_get_threshold(&mut self, grad_norm: f64) -> Option<f64> {
        // Store the new gradient norm
        if self.grad_norms.len() >= self.window_size {
            self.grad_norms.pop_front();
        }
        self.grad_norms.push_back(grad_norm);
    
    println!(
        "Stored Grad Norms: {}/{}",
        self.grad_norms.len(),
        self.window_size
    );

        // Avoid early clipping
        if self.grad_norms.len() < 5 {
            return None;
        }

        // Compute mean and standard deviation
        let mean: f64 = self.grad_norms.iter().sum::<f64>() / self.grad_norms.len() as f64;
        let variance: f64 = self.grad_norms.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.grad_norms.len() as f64;
        let std_dev = variance.sqrt();

        // Adaptive threshold
        Some(mean + self.factor * std_dev)
    }

    pub fn clip_gradients(&mut self, d_weights: &mut Matrix) {
        let norm = d_weights.norm(); // Ensure Matrix has a `norm()` method
println!("Gradient Norm: {:.6}", norm);
        if let Some(threshold) = self.update_and_get_threshold(norm) {
            println!("Adaptive Clipping | norm: {:.6}, threshold: {:.6}", norm, threshold);
            if norm > threshold {
                let scale = threshold / norm;
                d_weights.scale(scale); // Ensure Matrix has a `scale(f64)` method
            }
        }
    }

    pub fn add_weights(&mut self, weights: &Vec<f64>) {
        let norm: f64 = weights.iter().map(|&x| x * x).sum::<f64>().sqrt();
        self.grad_norms.push_back(norm);
    }

    pub fn reset_weights(&mut self) {
        self.grad_norms.clear();
    }

}

// Implement Default trait
impl Default for AdaptiveGradientClipping {
    fn default() -> Self {
        Self {
            window_size: 100, // Default moving window size
            factor: 2.0,      // Default factor for adaptive clipping
            grad_norms: VecDeque::with_capacity(100),
        }
    }
}
