use indicatif::{ProgressBar, ProgressStyle};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, Error as IoError};
use std::str::FromStr;
use serde_json::{from_reader, Value};
use crate::{loadr, log};
use matrix::matrix::{Dot, Matrix}; 
use activations::activations::{get_activation_and_derivative, ActivationTrait};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    Same,
    Valid,
}

impl FromStr for Padding {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "same" => Ok(Padding::Same),
            "valid" => Ok(Padding::Valid),
            _ => Err(format!("Invalid Padding type: {}", s)),
        }
    }
}

impl Padding {
    pub fn to_usize(&self) -> usize {
        match self {
            Padding::Same => 1,  // Keeps size the same
            Padding::Valid => 0, // No padding
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    Max,
    Average,
}

impl FromStr for PoolingType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "max" => Ok(PoolingType::Max),
            "average" => Ok(PoolingType::Average),
            _ => Err(format!("Invalid PoolingType: {}", s)),
        }
    }
}

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

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum WeightInitialization {
    Xavier,
    He,
}

impl FromStr for WeightInitialization {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "xavier" => Ok(WeightInitialization::Xavier),
            "he" => Ok(WeightInitialization::He),
            _ => Err(format!("Invalid WeightInitialization: {}", s)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Model {
    pub location: String,
    pub epochs: usize,
    pub check_points: usize,
    pub learning_rate: f64,
    pub logit_scaling_factor: f64,
    pub clipping: usize,
    pub clip_threshold: f64,
    pub temperature_scaling: f64,
    pub vocab_size: usize,
    pub batch_size: usize,
    pub num_classes: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_layer_scaling: usize,
    pub model_dimensions: usize,
    pub hidden_dimensions: usize,
    
    // CNN-Specific Parameters
    pub num_conv_layers: usize,
    pub conv_filters: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub stride_sizes: Vec<usize>,
    pub padding: Padding,
    
    // Pooling
    pub pooling_type: PoolingType,
    pub pooling_size: usize,
    pub pooling_stride: usize,
    
    // Fully Connected Layers
    pub num_dense_layers: usize,
    pub dense_units: Vec<usize>,
    
    // Regularization & Initialization
    pub dropout_rate: f64,
    pub weight_initialization: WeightInitialization,
    
    // Optimization
    pub optimizer: Optimizer,

     #[serde(skip)] // Prevent serialization issues
    pub activation_fn: Box<dyn ActivationTrait>,

    #[serde(skip)]
    pub derivative_fn: Box<dyn ActivationTrait>,

}
impl Default for Model {
    fn default() -> Self {
        let (activation_fn, derivative_fn) = get_activation_and_derivative("sigmoid".to_string(), 0.01, 1.0);

        Self {
            location: String::new(),
            epochs: 10,
            check_points: 1,
            learning_rate: 0.01,
            logit_scaling_factor: 1.0,
            clipping: 0,
            clip_threshold: 0.0,
            temperature_scaling: 1.0,
            vocab_size: 0,
            batch_size: 32,
            num_classes: 10,
            num_heads: 1,
            num_layers: 1,
            hidden_layer_scaling: 1,
            model_dimensions: 128,
            hidden_dimensions: 64,

            activation_fn,  
            derivative_fn,  

            // CNN Defaults
            num_conv_layers: 0,                // No convolutional layers by default
            conv_filters: vec![],              // No filters
            kernel_sizes: vec![],              // No kernels
            stride_sizes: vec![],              // No strides
            padding: Padding::Valid,           // Default to "valid" padding

            // Pooling Defaults
            pooling_type: PoolingType::Max,    // Default pooling type
            pooling_size: 2,                   // Typical default pooling size
            pooling_stride: 2,                 // Typical default stride

            // Dense Layer Defaults
            num_dense_layers: 2,               // A reasonable number of dense layers
            dense_units: vec![128, 64],        // Common default fully connected layers

            // Regularization & Initialization
            dropout_rate: 0.5,                 // No dropout by default
            weight_initialization: WeightInitialization::Xavier,  // Common initialization

            // Optimization
            optimizer: Optimizer::Adam,        // Adam is a common optimizer
        }
    }
}


impl Model {
    pub fn from_json(path: &str) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Read JSON as a generic `Value`
        let json_value: Value = from_reader(reader)?;

        // Log the JSON structure for debugging
        println!("Loaded JSON: {:#?}", json_value);

        // Extract individual fields, checking for missing values
        let location = json_value.get("location")
            .and_then(Value::as_str)
            .unwrap_or("./data/input")
            .to_string();

        let epochs = json_value.get("epochs")
            .and_then(Value::as_u64)
            .unwrap_or(10) as usize;

        let check_points = json_value.get("check_points")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;

        let learning_rate = json_value.get("learning_rate")
            .and_then(Value::as_f64)
            .unwrap_or(0.01);

        let logit_scaling_factor = json_value.get("logit_scaling_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);

        let clipping = json_value.get("clipping")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;

        let clip_threshold = json_value.get("clip_threshold")
            .and_then(Value::as_f64)
            .unwrap_or(0.125);

        let temperature_scaling = json_value.get("temperature_scaling")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);

        let vocab_size = json_value.get("vocab_size")
            .and_then(Value::as_u64)
            .unwrap_or(10000) as usize;

        let batch_size = json_value.get("batch_size")
            .and_then(Value::as_u64)
            .unwrap_or(64) as usize;

        let num_classes = json_value.get("num_classes")
            .and_then(Value::as_u64)
            .unwrap_or(10) as usize;

        let num_heads = json_value.get("num_heads")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;

        let num_layers = json_value.get("num_layers")
            .and_then(Value::as_u64)
            .unwrap_or(6) as usize;

        let hidden_layer_scaling = json_value.get("hidden_layer_scaling")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;

        let model_dimensions = json_value.get("model_dimensions")
            .and_then(Value::as_u64)
            .unwrap_or(784) as usize;

        let hidden_dimensions = json_value.get("hidden_dimensions")
            .and_then(Value::as_u64)
            .unwrap_or(2048) as usize;

        let num_conv_layers =  json_value.get("num_conv_layers")
            .and_then(Value::as_u64)
            .unwrap_or(2048) as usize;

        let optimizer = json_value.get("optimizer")
            .and_then(Value::as_str)
            .map(|s| Optimizer::from_str(s).unwrap_or(Optimizer::Adam))
            .unwrap_or(Optimizer::Adam);

        let weight_initialization = json_value.get("weight_initialization")
            .and_then(Value::as_str)
            .map(|s| WeightInitialization::from_str(s).unwrap_or(WeightInitialization::Xavier))
            .unwrap_or(WeightInitialization::Xavier);

        let pooling_type = json_value.get("pooling_type")
            .and_then(Value::as_str)
            .map(|s| PoolingType::from_str(s).unwrap_or(PoolingType::Max))
            .unwrap_or(PoolingType::Max);

        let padding = json_value.get("padding")
            .and_then(Value::as_str)
            .map(|s| Padding::from_str(s).unwrap_or(Padding::Same))
            .unwrap_or(Padding::Same);

        let pooling_size =  json_value.get("pooling_size")
            .and_then(Value::as_u64)
            .unwrap_or(2048) as usize;

        let pooling_stride =  json_value.get("pooling_stride")
            .and_then(Value::as_u64)
            .unwrap_or(2048) as usize;

        let num_dense_layers =  json_value.get("num_dense_layers")
            .and_then(Value::as_u64)
            .unwrap_or(2048) as usize;

        let dropout_rate =  json_value.get("dropout_rate")
            .and_then(Value::as_f64)
            .unwrap_or(0.5) as f64;

        let conv_filters = json_value.get("conv_filters")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_else(|| vec![32, 64, 128]); // Default values for CNN layers

        let kernel_sizes = json_value.get("kernel_sizes")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_else(|| vec![3, 3, 3]); // Default kernel sizes

        let stride_sizes = json_value.get("stride_sizes")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_else(|| vec![1, 1, 1]); // Default strides

        let dense_units = json_value.get("dense_units")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_else(|| vec![128, 64]); // Default values for fully connected layers
        
        let activation_fn_name = json_value.get("activation_fn_name")
            .and_then(Value::as_str)
            .unwrap_or("relu")
            .to_string();

        let alpha = json_value.get("alpha")
            .and_then(Value::as_f64)
            .unwrap_or(0.01); // Default for LeakyReLU/ELU/PReLU

        let lambda = json_value.get("lambda")
            .and_then(Value::as_f64)
            .unwrap_or(1.0); // Default for SELU


        let (activation_fn, derivative_fn) = get_activation_and_derivative(activation_fn_name.clone(), alpha, lambda);
        // Return parsed config as `Model`
        let model = Self {
            location,
            epochs,
            check_points,
            learning_rate,
            logit_scaling_factor,
            clipping,
            clip_threshold,
            temperature_scaling,
            vocab_size,
            batch_size,
            num_classes,
            num_heads,
            num_layers,
            hidden_layer_scaling,
            model_dimensions,
            hidden_dimensions,
            num_conv_layers,
            conv_filters,
            kernel_sizes,
            stride_sizes,
            padding,
            pooling_type,
            pooling_size,
            pooling_stride,
            num_dense_layers,
            dense_units,
            dropout_rate,
            weight_initialization,
            optimizer,
            activation_fn,
            derivative_fn
        };
        Ok(model)
    }


    pub fn train(&self) {
        println!("Training...");
        let location = self.location.as_str();
        let log_location = "./logs/log.csv";
        let squelch = false;
        let (training, training_labels, validation, validation_labels) = loadr::populate(location);

        log::matrix_stats(0, 0, &validation, log_location, "validation", squelch);
        log::n_elements("labels", &validation_labels, 10, log_location);

        let epochs = self.epochs;
        let batch_size = self.batch_size;
        //Debugging
        // let rows = shuffled_training.rows;
        let rows = 240;

        let total_batches = epochs * (rows / batch_size); // Total steps

        // Create a progress bar
        let pb = ProgressBar::new(total_batches as u64);
        pb.set_style(ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        //iterate epochs times
        for epoch in 0..epochs {

            let (
                shuffled_training, 
                shuffled_labels ) = Matrix::shuffled(&training, &training_labels);

            let mut batch_number = 0;
            for batch_start in (0..rows).step_by(batch_size){
                let batch_end = (batch_start + batch_size).min(shuffled_training.rows);
                // Extract a batch from the shuffled training data
                let training_batch = shuffled_training.slice(batch_start, batch_end);
                let label_batch = shuffled_labels[batch_start..batch_end].to_vec(); // Extract labels
                log::n_elements("label batch", &label_batch, 10, log_location);

                let output = self.forward(&training_batch);

                log::matrix_stats(epoch, batch_number, &output, &log_location, "output", squelch);

                batch_number += 1;
                pb.inc(1);
            }

        }

        pb.finish_with_message("Training Complete");
        
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {

        let mut output = input.clone();

        // Apply Convolutional Layers
        for (i, &num_filters) in self.conv_filters.iter().enumerate() {

            output = self.apply_convolution(
                &output, 
                num_filters, 
                self.kernel_sizes[i], 
                self.stride_sizes[i]
            );

            output = self.apply_activation(&output);
            
            // Apply Pooling
            output = self.apply_pooling(
                &output, 
                self.pooling_type, 
                self.pooling_size, 
                self.pooling_stride
            );
        }

        // Apply Fully Connected (Dense) Layers
        let mut dense_output = output.clone();

        for &units in &self.dense_units {
            dense_output = self.apply_dense(&dense_output, units);
            dense_output = self.apply_activation(&dense_output);
        }

        //return final output (logits)
        dense_output

    }

    // apply_pooling
    fn apply_pooling(&self, input: &Matrix, pooling_type:PoolingType, pool_size:usize, pool_stride: usize) -> Matrix{
        match pooling_type {
            PoolingType::Max => input.max_pooling(pool_size, pool_stride),
            PoolingType::Average => input.avg_pooling(pool_size, pool_stride),
        }
    }

    //apply_convolution
    fn apply_convolution(&self, input: &Matrix, num_filters: usize, kernel_size: usize, stride_size: usize) -> Matrix {
        //convolve the matrix
        let conv_result = input.convolve(
            &Matrix::random(kernel_size, kernel_size),
            stride_size,
            self.padding.to_usize()
        );

        let output_rows = conv_result.rows;  
        let output_cols = conv_result.cols;  

        let mut output_data = vec![0.0; output_rows * output_cols * num_filters];

        for filter_idx in 0..num_filters {
            let kernel = Matrix::random(kernel_size, kernel_size);
            let conv_result = input.convolve(&kernel, stride_size, self.padding.to_usize());

            // Ensure sizes match before copying
            if conv_result.data.len() != output_rows * output_cols {
                panic!(
                    "Size mismatch! conv_result.data has {} elements, but expected {}",
                    conv_result.data.len(),
                    output_rows * output_cols
                );
            }
        
            let offset = filter_idx * output_rows * output_cols;
            output_data[offset..offset + output_rows * output_cols].copy_from_slice(&conv_result.data);
        }

        Matrix::new(output_rows * num_filters, output_cols, output_data)

    }


    // apply_activation
    fn apply_activation(&self, input: &Matrix) -> Matrix {
        input.map(|x| self.activation_fn.apply(x)) 
    }

    fn apply_dense(&self, input: &Matrix, units: usize) -> Matrix {
        // Generate weight and bias matrices
        let weights = Matrix::random(input.cols, units); // Using your `Matrix::random`
        let bias = Matrix::zeros(input.rows, units); // Adjust bias to match output rows

        // 🚀 Perform matrix multiplication using your `Dot` trait
        let output = input.dot(&weights) + bias; // Bias is now correctly sized

        output
    }

}
