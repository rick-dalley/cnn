use std::io::Write;
use std::fs::OpenOptions;
use std::sync::Once;
use matrix::matrix::Matrix;
use std::process;
use crate::cnn::Model;


/// Enum to define logging output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogTo {
    Screen,
    JSON,
    Log,
}

pub fn memory_usage(tag: &str) {
    let pid = process::id();
    let output = std::process::Command::new("ps")
        .arg("-o")
        .arg("rss=") // Resident Set Size (actual memory in KB)
        .arg("-p")
        .arg(format!("{}", pid))
        .output()
        .expect("Failed to get memory usage");

    if let Ok(mem_usage) = String::from_utf8(output.stdout) {
        println!("Memory Usage at {}: {} KB", tag, mem_usage.trim());
    }
}

// log training metrics
pub fn training_metrics(
    epoch: usize, 
    loss: f64, 
    accuracy: f64, 
    std: f64,
    mean: f64,
    log_location: &str,
    squelch: bool
) {

    if squelch {
        return;
    }
    static HEADER_PRINTED: Once = Once::new();
    
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    HEADER_PRINTED.call_once(|| {
        writeln!(file, "epoch, avg_loss, accuracy, final_output_weight_std, final_output_weight_mean, ff_output_weights_std, ff_output_weights_mean")
            .expect("Failed to write header.");
    });

    writeln!(file, "{}, {},{}, {}, {}", epoch, loss, accuracy, std, mean)
        .expect("Failed to write log.");

}

pub fn matrix_norms(epoch:usize, iteration:usize, matrix: Matrix, log_location:&str){
    // Create or open the log file to append
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");


        writeln!(file, "{}, {},{}", epoch, iteration, matrix.compute_norm()).expect("Could not write to file");

}

//log_matrix_stats
pub fn matrix_stats(epoch:usize, iteration:usize, matrix:&Matrix, log_location: &str, name:&str, squelch:bool) {
    if squelch {
        return;
    }
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");
        


    static HEADER_PRINTED: Once = Once::new();
    HEADER_PRINTED.call_once(|| {
        writeln!(file, "name, epoch, iteration, norm, mean, std, min, max").expect("Failed to write header.");
    });

    writeln!(file, "{}, {}, {}, {}, {}, {}, {}, {}", name, epoch, iteration, matrix.compute_norm(), matrix.mean(), matrix.std_dev(), matrix.min(), matrix.max()).expect("Could not write to file");

}

//log_n_elements
pub fn n_elements(name:&str, slice:&Vec<f64>, n_elements:usize, log_location: &str){

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    writeln!(file, "{} first {}, {:?}", name,n_elements, &slice[..n_elements.min(slice.len())] ).expect("Could not write to file");

}

// log_sample
pub fn sample(name: &str, rows:usize, num_elements:usize, matrix:&Matrix, log_location:&str, squelch:bool){
    if squelch {
        return;
    }
    for i in 0..rows.min(matrix.rows) {
        let row_slice = matrix.sample(i, num_elements); 
        n_elements(name, &row_slice, rows, log_location);
    }
}


/// Logs the model configuration based on the selected log type.
pub fn model(model: &Model, log_location: &str, log_to: LogTo) {
    match log_to {
        LogTo::Screen => {
            // Pretty-print to console
            println!("{:#?}", model);
        }
        LogTo::JSON => {
            // Serialize to JSON and save to a file
            let json_output = serde_json::to_string_pretty(&model)
                .expect("Failed to serialize model config");

            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(format!("{}.json", log_location))
                .expect("Failed to open log file");

            writeln!(file, "{}", json_output)
                .expect("Failed to write model config to log file.");
        }
        LogTo::Log => {
            static HEADER_PRINTED: Once = Once::new();

            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(format!("{}.csv", log_location))
                .expect("Failed to open log file");

            // Write CSV header only once
            HEADER_PRINTED.call_once(|| {
                writeln!(file, "model.location,
                model.epochs,
                model.check_points,
                model.learning_rate,
                model.logit_scaling_factor,
                model.clipping,
                model.clip_threshold,
                model.temperature_scaling,
                model.vocab_size,
                model.batch_size,
                model.num_classes,
                model.num_heads,
                model.num_layers,
                model.hidden_layer_scaling,
                model.model_dimensions,
                model.hidden_dimensions,
                model.num_conv_layers,
                model.conv_filters,
                model.kernel_sizes,
                model.stride_sizes,
                model.padding,
                model.pooling_type,
                model.pooling_size,
                model.pooling_stride,
                model.num_dense_layers,
                model.dense_units,
                model.dropout_rate,
                model.weight_initialization,
                model.optimizer")
                    .expect("Failed to write header.");
            });

            // Append model config as CSV row
            writeln!(
                file,
                "{}, {},{},{},{},{},{},{},{},{},{},{},{},{},{},{:?},{:?},{:?},{:?},{:?},{:?},{},{},{:?},{:?},{},{:?},{:?}",
                model.location,
                model.epochs,
                model.check_points,
                model.learning_rate,
                model.logit_scaling_factor,
                model.clipping,
                model.clip_threshold,
                model.temperature_scaling,
                model.batch_size,
                model.num_classes,
                model.num_heads,
                model.num_layers,
                model.hidden_layer_scaling,
                model.model_dimensions,
                model.hidden_dimensions,
                model.num_conv_layers,
                model.conv_filters,
                model.kernel_sizes,
                model.stride_sizes,
                model.padding,
                model.pooling_type,
                model.pooling_size,
                model.pooling_stride,
                model.num_dense_layers,
                model.dense_units,
                model.dropout_rate,
                model.weight_initialization,
                model.optimizer
            ).expect("Failed to write model config to log file.");
        }
    }
}