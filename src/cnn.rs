use indicatif::{ProgressBar, ProgressStyle};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, Error as IoError};
use std::str::FromStr;
use serde_json::{from_reader, Value};
use crate::{loadr, log, layr};
use matrix::matrix::{Dot, Matrix, FlatteningStrategy}; 


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
    pub flattening_strategy: FlatteningStrategy,
    // Fully Connected Layers
    pub num_dense_layers: usize,
    pub dense_units: Vec<usize>,
    
    // Regularization & Initialization
    pub dropout_rate: f64,
    pub weight_initialization: WeightInitialization,
    
    // optimization
    pub optimizer: Optimizer,

    // layers
    #[serde(skip)]
    pub dense_layers: Vec<layr::Layer>,
    #[serde(skip)]
    pub convolution_layers: Vec<layr::Layer>,

}

impl Default for Model {
    fn default() -> Self {
        Self {
            location: String::new(),
            epochs: 10,
            check_points: 1,
            learning_rate: 0.01,
            logit_scaling_factor: 1.0,
            clipping: 0,
            clip_threshold: 0.0,
            temperature_scaling: 1.0,
            batch_size: 32,
            num_classes: 10,
            num_heads: 1,
            num_layers: 1,
            hidden_layer_scaling: 1,
            model_dimensions: 128,
            hidden_dimensions: 64,
            
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
            flattening_strategy:FlatteningStrategy::MeanPooling,
            // Dense Layer Defaults
            num_dense_layers: 2,               // A reasonable number of dense layers
            dense_units: vec![128, 64],        // Common default fully connected layers

            // Regularization & Initialization
            dropout_rate: 0.5,                 // No dropout by default
            weight_initialization: WeightInitialization::Xavier,  // Common initialization

            // Optimization
            optimizer: Optimizer::Adam,        // Adam is a common optimizer
            dense_layers:Vec::new(),
            convolution_layers:Vec::new(),

        }
    }
}


impl Model {

    //initialize from json
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

        let flattening_strategy_name = json_value.get("flattening_strategy")
            .and_then(Value::as_str)
            .unwrap_or("mean_pooling")
            .to_string();

        let flattening_strategy = FlatteningStrategy::from_str(&flattening_strategy_name);

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
        
            
        //prepare the layers
        let layer_function_strategy = json_value.get("layer_function_strategy")
            .and_then(Value::as_str)
            .unwrap_or("default")
            .to_string();

        let approach = json_value.get("approach")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| "default".to_string());  // Provide a sensible default

        //select the activation function
        let dense_activation_fn = json_value.get("dense_activation_fn")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| "relu".to_string());  // Default to "relu"

        let conv_activation_fn = json_value.get("conv_activation_fn")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| "relu".to_string());  // Default to "relu"

        let alpha = json_value.get("alpha")
            .and_then(Value::as_f64)
            .unwrap_or(0.01); // Default for LeakyReLU/ELU/PReLU

        let lambda = json_value.get("lambda")
            .and_then(Value::as_f64)
            .unwrap_or(1.0); // Default for SELU

        let (dense_family, conv_family) = match layer_function_strategy.as_str() {
            "default" => ("relu", "relu"),
            "approach" => match approach.as_str() {
                "resnet" => ("relu","relu"), 
                "vgg" => ("tanh","tanh"),
                _ => panic!("Unknown approach: {}", approach),
            },
            "custom" => (dense_activation_fn.as_str(), conv_activation_fn.as_str()),
            _ => panic!("Unknown layer function strategy: {}", layer_function_strategy),
        };

        let dense_layers = Self::initialize_dense_layers(
            &dense_units, 
            dense_family, 
            alpha, 
            lambda
        );

        let convolution_layers = Self::initialize_convolutional_layers(
            &conv_filters, 
            &kernel_sizes, 
            &stride_sizes, 
            conv_family, 
            alpha, 
            lambda
        );

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
            flattening_strategy,
            num_dense_layers,
            dense_units,
            dropout_rate,
            weight_initialization,
            optimizer,
            dense_layers,
            convolution_layers,
        };

        Ok(model)
    }

    pub fn initialize_dense_layers(
        dense_units: &[usize], 
        dense_family: &str, 
        alpha: f64, 
        lambda: f64
    ) -> Vec<layr::Layer> {
        let mut layers = Vec::new();

        for &_units in dense_units {
            layers.push(layr::Layer::new_dense(dense_family.to_string(), alpha, lambda));
        }

        layers
    }

    pub fn initialize_convolutional_layers(
        conv_filters: &[usize], 
        kernel_sizes: &[usize], 
        stride_sizes: &[usize], 
        conv_family: &str, 
        alpha: f64, 
        lambda: f64
    ) -> Vec<layr::Layer> {
        let mut layers = Vec::new();

        for (i, &filters) in conv_filters.iter().enumerate() {
            let kernel_size = kernel_sizes.get(i).copied().unwrap_or(3); // Default: 3x3 kernels
            let stride = stride_sizes.get(i).copied().unwrap_or(1); // Default stride: 1

            layers.push(layr::Layer::new_convolutional(
                filters, 
                kernel_size,  
                stride,        
                conv_family.to_string(),
                alpha,
                lambda
            ));
        }

        layers
    }

    pub fn train(&mut self) {
        println!("Training...");
        let location = self.location.as_str();
        let log_location = "./logs/log.csv";
        let squelch = false;
        let (training, training_labels, validation, validation_labels) = loadr::populate(location);

        log::matrix_stats(0, 0, &validation, log_location, "validation", squelch);

        let epochs = self.epochs;
        let batch_size = self.batch_size;
        //Debugging
        // let rows = shuffled_training.rows;
        let rows = 32;

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
println!("Getting the batch");
                // Getting batch
                let batch_end = (batch_start + batch_size).min(shuffled_training.rows);
                let batch_data = shuffled_training.slice(batch_start, batch_end);
                let batch_labels = shuffled_labels[batch_start..batch_end].to_vec(); // Extract labels

println!("performing forward");
                // predict
                let predictions = self.forward(&batch_data);

println!("calculating loss");
                //calculate loss
                let loss = self.calculate_loss(&predictions, &batch_labels);

println!("performing backward");
                //perform backward propagation
                self.backward(&predictions, &batch_labels, &batch_data);

println!("updating weights");
                //upate weights
                self.update_weights();

                log::matrix_stats(epoch, batch_number, &predictions, &log_location, "output", squelch);

                batch_number += 1;
                pb.inc(1);
            }

        }

        pb.finish_with_message("Training Complete");
        
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {

        // Apply Convolutional Layers
       let mut feature_maps = vec![input.clone()]; // Start with the input matrix as the first feature map

        let mut layer_count = 1;
        println!("Applying convolutions");

        for layer in &self.convolution_layers {
            println!("Applying convolutions Layer:{}", layer_count);

            let mut new_feature_maps = Vec::new();
            
            // Process only the current feature maps and replace them
            for feature_map in &feature_maps {
                let conv_maps = self.apply_convolution(feature_map, layer);
                new_feature_maps = conv_maps; // Direct replacement instead of extend
            }

            feature_maps = new_feature_maps; // Keep only the latest feature maps
            layer_count += 1;
        }

        println!("Applying pooling to {} layers on feature_maps of len:{}", layer_count, feature_maps.len());
        // Apply Pooling
        let pooled_maps = self.apply_pooling(
            &feature_maps, 
            self.pooling_type, 
            self.pooling_size, 
            self.pooling_stride
        );

        // Get the expected input size for the first dense layer
        let expected_dense_input_size = self.dense_units.first().copied().unwrap_or(512);

        println!("Flattening the feature map - expected_dense_input_size:{}", expected_dense_input_size,);
        // Apply the selected flattening strategy
        let flattened_features = match self.flattening_strategy {
            FlatteningStrategy::MeanPooling => Matrix::mean_pooling_flatten(pooled_maps, expected_dense_input_size),
            FlatteningStrategy::Strided => Matrix::strided_flatten(pooled_maps, expected_dense_input_size),
            FlatteningStrategy::Convolution => Matrix::convolution_flatten(pooled_maps, expected_dense_input_size),
        };
        
        println!("flattened_features - rows:{} x {} cols", flattened_features.rows, flattened_features.cols);
   
        for (i, layer) in self.dense_layers.iter_mut().enumerate() {
            if layer.weights.rows == 0 || layer.weights.cols == 0 {
                let input_dim = if i == 0 {
                    flattened_features.cols // First dense layer takes the flattened feature map
                } else {
                    self.dense_units[i - 1] // Subsequent layers use the previous layer's output
                };

                let output_dim = self.dense_units[i]; // Get the output size from config

                layer.weights = Matrix::random(input_dim, output_dim);
                layer.biases = Matrix::zeros(1, output_dim);

                println!(
                    "Initialized Dense Layer {}: weights {}x{}, biases {}x{}",
                    i, layer.weights.rows, layer.weights.cols, layer.biases.rows, layer.biases.cols
                );
            }
        }

 
        println!("Applying dense");
        let mut output = flattened_features.clone();

        // Apply Fully Connected (Dense) Layers
        for layer in &self.dense_layers {
            output = self.apply_dense(&output, &layer);
        }

        // Return final output (logits)
        output
    }

    // apply pooling
    fn apply_pooling(&self, 
        feature_maps: &Vec<Matrix>, 
        pooling_type: PoolingType, 
        pool_size: usize, 
        pool_stride: usize) -> Vec<Matrix> {

        let pooled_maps: Vec<Matrix> = feature_maps.iter()
            .map(|feature_map| match pooling_type {
                PoolingType::Max => feature_map.max_pooling(pool_size, pool_stride),
                PoolingType::Average => feature_map.avg_pooling(pool_size, pool_stride),
            })
            .collect();

        pooled_maps
    }

    // apply convolution
    fn apply_convolution(&self, input: &Matrix, layer: &layr::Layer) -> Vec<Matrix> {
        if let layr::LayerType::Convolutional { stride_size, kernel_size } = layer.layer_type {
            let num_filters = layer.weights.rows; // Number of filters
            let padding = self.padding.to_usize();

            let output_rows = (input.rows - kernel_size + 2 * padding) / stride_size + 1;
            let output_cols = (input.cols - kernel_size + 2 * padding) / stride_size + 1;

            let mut feature_maps = Vec::with_capacity(num_filters);

            // Initialize feature maps for each filter
            for _ in 0..num_filters {
                feature_maps.push(Matrix::zeros(output_rows, output_cols));
            }

            // Slide the kernel across the image
            for row in (0..input.rows - kernel_size).step_by(stride_size) {
                for col in (0..input.cols - kernel_size).step_by(stride_size) {
                    // Extract the kernel-sized patch from the input
                    let patch = match input.get_filter(row, col, kernel_size) {
                        Ok(patch) => patch,
                        Err(err) => panic!("Error extracting filter: {}", err),
                    };

                    // Apply each filter to the patch
                    for filter_idx in 0..num_filters {
                        let filter_row = layer.weights.slice(filter_idx, filter_idx + 1).transpose();
                        let bias = layer.biases.data[filter_idx];

                        let conv_value = patch.dot(&filter_row).data[0] + bias;

                        // Determine output index
                        let output_row_idx = row / stride_size;
                        let output_col_idx = col / stride_size;

                        feature_maps[filter_idx].data[output_row_idx * output_cols + output_col_idx] = conv_value;
                    }
                }
            }

            feature_maps
        } else {
            panic!("apply_convolution() called on a non-convolutional layer!");
        }
    }

    fn apply_dense(&self, output: &Matrix, layer: &layr::Layer) -> Matrix {
        if let layr::LayerType::Dense = layer.layer_type {

            println!(
        "apply_dense Debug: output shape = {}x{}, layer.weights shape = rows:{} x cols:{}, layer.biases shape = {}x{}",
        output.rows, output.cols,
        layer.weights.rows, layer.weights.cols,
        layer.biases.rows, layer.biases.cols
        );
            let mut result = output.dot(&layer.weights) + &layer.biases;
            result = result.apply(&|x| layer.activation_fn.apply(x));
            result
        } else {
            panic!("Dense layer processing called on a non-dense layer!");
        }
    }

    fn calculate_loss(&self, predictions: &Matrix, labels: &[usize]) -> f64 {
        let epsilon = 1e-9; // To avoid log(0)
        
        // Convert labels to one-hot matrix
        let one_hot_labels = Matrix::from_labels(labels, predictions.cols);

        // Ensure the predictions have valid probability values
        let clipped_predictions = predictions.clamp_to(epsilon, 1.0 - epsilon);

        // Compute cross-entropy loss: -sum(y_true * log(y_pred))
        let log_preds = clipped_predictions.apply(&|x:f64| x.ln());
        let loss_matrix = &one_hot_labels * &log_preds; // Element-wise multiplication
        let loss = -loss_matrix.sum() / labels.len() as f64; // Normalize by batch size

        loss
    }

    fn backward(
        &mut self, 
        predictions: &Matrix, 
        batch_labels: &Vec<usize>, 
        batch_data: &Matrix
    ) {
        // Convert labels into one-hot matrix
        let batch_labels_matrix = Matrix::from_labels(batch_labels, predictions.cols);

        // Compute dL/dO (Cross-Entropy Loss Gradient)
        let d_loss = predictions - &batch_labels_matrix;

        // Backpropagation through dense layers (iterate in reverse)
        let mut d_output = d_loss.clone();
        
        for layer in self.dense_layers.iter_mut().rev() {
            let d_activation = d_output.apply(&|x| layer.derivative_fn.apply(x));
            let d_weights = batch_data.transpose().dot(&d_activation);
            let d_biases = d_activation.sum_axis(matrix::matrix::Orientation::ColumnWise);

            // directly update weights and biases
            layer.weights = &layer.weights - &(d_weights * self.learning_rate);
            layer.biases = &layer.biases - &(d_biases * self.learning_rate);

            // Propagate error backward
            d_output = d_activation.dot(&layer.weights.transpose());
        }

        // Backpropagation through convolutional layers (iterate in reverse)
        let mut conv_gradients = Vec::new();
        for layer in self.convolution_layers.iter_mut().rev() {
            let d_activation = d_output.apply(&|x| layer.derivative_fn.apply(x));
            let d_filter = batch_data.convolve(&d_activation, 1, 0); // Assuming `convolve` exists

            // Store convolutional gradients
            conv_gradients.push(d_filter.clone());

            //directly update weights and biases
            layer.weights = &layer.weights - &(d_filter * self.learning_rate);
            layer.biases = &layer.biases - &(d_activation.sum_axis(matrix::matrix::Orientation::ColumnWise) * self.learning_rate);

            // Propagate error backward
            d_output = d_activation.dot(&layer.weights.transpose());
        }
    }

    fn update_weights(&self) {

    }

}
