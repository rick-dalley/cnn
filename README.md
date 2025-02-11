# Layer Function Strategy

The `layer_function_strategy` setting determines how activation, loss, and backpropagation functions are selected for training.

## Options:

| Strategy     | Description                                                                          |
| ------------ | ------------------------------------------------------------------------------------ |
| `"default"`  | Uses standard best-practice functions for CNN training.                              |
| `"approach"` | Selects functions based on a pre-defined architecture (e.g., ResNet, LeNet).         |
| `"custom"`   | Allows manual selection of forward propagation, loss, and backpropagation functions. |

## Configuration Example

### Default Strategy (Best Practices)

```json
"layer_function_strategy": "default"
```

### Approach-Based Strategy

```json
"layer_function_strategy": "approach",
"approach": "resnet"
```

### Custom Strategy

```json
"layer_function_strategy": "custom",
"forward_prop_fn": "sigmoid",
"loss_fn": "cross_entropy",
"backward_prop_fn": "sigmoid_derivative"
```

## Supported Approaches

| Approach         | Forward Function | Loss Function                 |
| ---------------- | ---------------- | ----------------------------- |
| `"resnet"`       | `"relu"`         | `"cross_entropy"`             |
| `"efficientnet"` | `"swish"`        | `"categorical_cross_entropy"` |
| `"lenet"`        | `"tanh"`         | `"mse"`                       |

If `"custom"` is selected, the following functions can be specified:

- **`forward_prop_fn`**: `"sigmoid"`, `"relu"`, `"tanh"`, `"swish"`, etc.
- **`loss_fn`**: `"cross_entropy"`, `"categorical_cross_entropy"`, `"mse"`, etc.
- **`backward_prop_fn`**: Corresponding derivatives like `"sigmoid_derivative"`, `"relu_derivative"`, etc.
