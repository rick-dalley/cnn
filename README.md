# Convolutional Neural Network (CNN) Implementation in Rust

## Overview

This project implements a Convolutional Neural Network (CNN) in Rust from scratch. It includes custom matrix operations, activation functions, pooling layers, and a configurable training pipeline.

## Requirements

- Rust toolchain installed
- Dependencies as specified in `Cargo.toml`

## Configuration

The network is configured via a `config.json` file with the following parameters:

```json
{
  "location": "./data/input",
  "epochs": 2,
  "check_points": 0,
  "learning_rate": 0.1,
  "logit_scaling_factor": 1.0,
  "clipping": 2,
  "clip_threshold": 0.125,
  "temperature_scaling": 1.0,
  "batch_size": 16,
  "num_classes": 10,
  "num_layers": 6,
  "layer_function_strategy": "default",
  "approach": "resnet",
  "dense_activation_fn": "sigmoid",
  "conv_activation_fn": "tanh",
  "alpha": 1.6732632423543772,
  "lambda": 1.0507009873554802,
  "hidden_dimensions": 1024,
  "num_conv_layers": 3,
  "conv_filters": [32, 64, 128],
  "kernel_sizes": [4, 4, 4],
  "stride_sizes": [1, 1, 1],
  "padding": "same",
  "pooling_type": "max",
  "pooling_size": 2,
  "pooling_stride": 2,
  "num_dense_layers": 1,
  "dense_units": [512],
  "dropout_rate": 0.5,
  "weight_initialization": "xavier",
  "optimizer": "adam"
}
```

## Important Considerations

### Kernel Size Constraints

- The **kernel size** must divide evenly into the image size. If this condition is not met, the program will panic.

### Batch Size Constraints

- The **batch size** must be at least **twice the stride size**. For example, if the filter size is `4`, then the `batch_size` must be at least `8`.

## Running the Model

To train the CNN, execute the following command:

```sh
cargo run --release
```

For debugging:

```sh
cargo run
```

# Layer Function Strategy

The `layer_function_strategy` setting determines how activation functions are assigned to **convolutional** and **dense** layers.

## Available Strategies:

| Strategy     | Description                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------- |
| `"default"`  | Uses `"relu"` as the activation function for both convolutional and dense layers.                   |
| `"approach"` | Assigns activation functions based on a predefined neural network architecture (e.g., ResNet, VGG). |
| `"custom"`   | Allows manual selection of activation functions for convolutional and dense layers.                 |

---

## Configuration Examples

### Default Strategy (Best Practices)

Uses `"relu"` for both convolutional and dense layers.

```json
"layer_function_strategy": "default"
```

---

### Approach-Based Strategy

Automatically sets activation functions based on the chosen architecture.

```json
"layer_function_strategy": "approach",
"approach": "resnet"
```

| Approach   | Convolutional Activation | Dense Activation |
| ---------- | ------------------------ | ---------------- |
| `"resnet"` | `"relu"`                 | `"relu"`         |
| `"vgg"`    | `"tanh"`                 | `"tanh"`         |

---

### Custom Strategy

Manually specify different activation functions for **convolutional** and **dense** layers.

```json
"layer_function_strategy": "custom",
"conv_activation_fn": "swish",
"dense_activation_fn": "sigmoid"
```

- **`conv_activation_fn`**: The activation function for convolutional layers.
- **`dense_activation_fn`**: The activation function for dense layers.

### Supported Activation Functions

- `"relu"`
- `"sigmoid"`
- `"tanh"`
- `"swish"`
- `"leaky_relu"`
- `"softmax"`
- `"selu"`
- `"elu"`

---

## How It Works in the Code

- If `"default"` is selected → both convolutional and dense layers use `"relu"`.
- If `"approach"` is selected → activation functions are set based on the selected architecture.
- If `"custom"` is selected → explicitly defined activation functions are applied.

This ensures flexibility while maintaining a structured way to configure layer functions.

## Performance Considerations

- Debug mode is significantly slower due to additional checks.
- Optimizing matrix operations for parallel execution may improve performance in future updates.
- Running on a machine with an **NVIDIA GPU** would be more efficient than a Mac for large-scale training.

## Future Work

- Implement parallel computation for matrix operations.
- Optimize memory usage for large batch sizes.
- Explore GPU acceleration options.

## License

This project is released under the MIT License.
