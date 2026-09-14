# `src/train.py` — build and train the network

## Goal

This file turns prepared train/test CSV files into a saved ReLU-plus-softmax
model. It also creates the loss/accuracy chart.

## Functions

| Function | Takes | Returns / does | Why it helps |
| --- | --- | --- | --- |
| `create_network(hidden_layers, input_features, seed=42)` | Hidden sizes, input feature count, seed | Built `MultilayerPerceptron` | Creates ReLU layers for every hidden size and a final two-neuron softmax layer. |
| `plot_learning_curves(history, save_path)` | Model history and image path | Saves a PNG | Shows whether categorical loss falls and accuracy rises across epochs. |
| `train_model(...)` | Data paths, layer sizes, training settings, output paths, seed, optional scaler path | Trained model | Coordinates loading, training, evaluation, plotting, and model saving. |

## `train_model` inputs

| Input | Meaning |
| --- | --- |
| `train_data_path`, `test_data_path` | Prepared CSV files created by `split_dataset`. |
| `hidden_layers` | List such as `[24, 24]`. |
| `epochs` | How many complete passes over training rows to make. |
| `batch_size` | How many rows produce one weight update. |
| `learning_rate` | Size of every gradient-descent step. |
| `model_out`, `plot_out` | Paths for the saved JSON model and chart image. |
| `seed` | Makes initial weights and mini-batch order repeatable. |
| `scaler_path` | Optional explicit training scaler; it is saved inside the model. |

## Result

The function prints test metrics, saves the learning-curve image, saves the
model JSON, and returns the trained `MultilayerPerceptron` for Python code.
