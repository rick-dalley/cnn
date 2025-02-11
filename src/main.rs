pub mod cnn;
pub mod log;
pub mod loadr;

use std::process;
use cnn::Model; // Ensure `cnn_model` is the correct module path


fn main() {
    let config_path = "./config/config.json";

    // Load CNN model from JSON
    let cnn_model = match Model::from_json(config_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading CNN model: {}", e);
            process::exit(1);
        }
    };

    // Log CNN model to the screen
    log::model(&cnn_model, "", log::LogTo::Screen);

    cnn_model.train();
}