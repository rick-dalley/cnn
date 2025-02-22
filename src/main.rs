pub mod cnn;
pub mod log;
pub mod loadr;
pub mod layr;
pub mod clipr;
pub mod optimizr;

use std::process;
use cnn::Model; // ensure `cnn_model` is the correct module path


fn main() {
    let config_path = "./config/config.json";

    // load the CNN model configuration from JSON
    let mut cnn_model = match Model::from_json(config_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading CNN model: {}", e);
            process::exit(1);
        }
    };
    // cnn_model.print_summary();

    //train the model
    cnn_model.train();
}