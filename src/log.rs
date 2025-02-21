use std::io::Write;
use std::fs::{self, OpenOptions};
use std::fmt::Debug;
use std::path::Path;
use std::sync::Once;
use matrix::Matrix;
use std::process;


/// enum to define logging output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogTo {
    Screen,
    JSON,
    Log,
}

pub fn clear_logs(dir_path: &str) -> std::io::Result<()> {
    let path = Path::new(dir_path);

    if path.exists() && path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            
            if entry_path.is_file() {
                fs::remove_file(&entry_path)?;
            } else if entry_path.is_dir() {
                fs::remove_dir_all(&entry_path)?; // Remove subdirectories and their contents
            }
        }
    }
    Ok(())
}


pub fn memory_usage(tag: &str) {
    let pid = process::id();
    let output = std::process::Command::new("ps")
        .arg("-o")
        .arg("rss=") // resident Set Size (actual memory in KB)
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
    
    // create or open the log file to append
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
    // create or open the log file to append
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

pub fn n_elements<T: Debug>(name: &str, slice: &[T], n_elements: usize, log_location: &str) {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_location)
        .expect("Failed to open log file");

    writeln!(file, "{} first {}, {:?}", name, n_elements, &slice[..n_elements.min(slice.len())])
        .expect("Could not write to file");
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

