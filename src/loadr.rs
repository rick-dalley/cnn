use std::fs::File;
use std::io::{BufReader, BufRead};
use matrix::matrix::Matrix;

/// Loads the training & validation data into `Matrix` objects.
pub fn populate(location: &str) -> (Matrix, Vec<f64>, Matrix, Vec<f64>) {
    let training_data_path = format!("{}/training_data.csv", location);
    let training_labels_path = format!("{}/training_labels.json", location);
    let validation_data_path = format!("{}/validation_data.csv", location);
    let validation_labels_path = format!("{}/validation_labels.json", location);

    let training_data = load_csv(&training_data_path);
    let training_labels = load_labels(&training_labels_path);
    let validation_data = load_csv(&validation_data_path);
    let validation_labels = load_labels(&validation_labels_path);

    (training_data, training_labels, validation_data, validation_labels)
}

/// Reads a CSV file into a `Matrix`
fn load_csv(path: &str) -> Matrix {
    let file = File::open(path).expect(&format!("Failed to open CSV file: {}", path));
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    let mut num_rows = 0;
    let mut num_cols = None;

    for line in reader.lines() {
        let row: Vec<f64> = line
            .expect("Failed to read line")
            .split(',')
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        if num_cols.is_none() {
            num_cols = Some(row.len());
        } else if row.len() != num_cols.unwrap() {
            panic!(
                "Inconsistent column count in CSV: expected {}, found {} in file {}",
                num_cols.unwrap(),
                row.len(),
                path
            );
        }

        data.extend(row);
        num_rows += 1;
    }

    Matrix::new(num_rows, num_cols.unwrap(), data)
}

/// Reads a JSON file containing labels into a `Vec<f64>`
fn load_labels(path: &str) -> Vec<f64> {
    let file = File::open(path).expect(&format!("Failed to open JSON file: {}", path));
    let reader = BufReader::new(file);
    
    serde_json::from_reader(reader).expect(&format!("Failed to parse JSON labels from {}", path))
}