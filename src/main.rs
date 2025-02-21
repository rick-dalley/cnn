pub mod cnn;
pub mod log;
pub mod loadr;
pub mod layr;
pub mod clipr;

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
    cnn_model.print_summary();
    // uncommenbt tge next line log the CNN model to the screen
    // log::model(&cnn_model, "", log::LogTo::Screen);

    //train the model
    cnn_model.train();
}