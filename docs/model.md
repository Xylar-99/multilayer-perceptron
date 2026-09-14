# `src/model.py` — neural-network mathematics

## Goal

This file contains the learnable parts of the project. It deliberately keeps
the forward pass, gradients, and weight updates visible rather than hiding
them behind a framework.

## Output classes

The final layer has two softmax neurons:

```text
output[0] = P(benign)
output[1] = P(malignant)
P(benign) + P(malignant) = 1
```

The model predicts the index with the largest probability: `0` for benign or
`1` for malignant.

## Module functions

| Function | Takes | Returns | Purpose |
| --- | --- | --- | --- |
| `relu(values)` | Layer scores | Same-shaped values | Keeps positive scores and changes negative scores to zero. |
| `softmax(values)` | One score row per sample | Two probabilities per sample | Converts final scores into probabilities that sum to one. |
| `categorical_cross_entropy(targets, probabilities)` | One-hot true labels and softmax output | Average loss number | Measures how far predictions are from the correct class. |
| `class_labels(labels)` | `B`/`M` or `0`/`1` labels | Integer class indexes | Uses `B → 0` and `M → 1`. |
| `one_hot(labels)` | Class labels | Rows like `[1, 0]` or `[0, 1]` | Gives categorical loss one target position per output neuron. |
| `classification_metrics(y_true, predictions)` | True and predicted class indexes | Accuracy, precision, recall, confusion matrix | Summarises binary classification quality. |

## `DenseLayer`

Each `DenseLayer` stores one weight matrix `W`, one bias row `b`, its most
recent inputs, activations, and gradients.

| Method | Takes | Returns / changes | Purpose |
| --- | --- | --- | --- |
| `DenseLayer(units, activation)` | Neuron count and `relu` or `softmax` | New layer | Defines the layer shape and activation. |
| `build(input_features, rng=None)` | Previous-layer width and random generator | Initializes `W` and `b` | Uses He initialization for ReLU and Xavier initialization for softmax. |
| `forward(inputs)` | Batch matrix | Activations `A` | Calculates `Z = inputs @ W + b`, then applies activation. |
| `backward(gradient, output_layer=False)` | Gradient from the next layer | Gradient for the previous layer; stores `dW`, `db` | Calculates parameter gradients for one mini-batch. |
| `update(learning_rate)` | Learning rate | Changes `W` and `b` | Applies gradient descent. |
| `from_saved_data(saved_layer)` | One layer dictionary from model JSON | Restored `DenseLayer` | Recreates a saved layer without randomizing it. |

For every layer, the important equations are:

```text
Z = inputs @ W + b
dW = inputs.T @ dZ / batch_size
db = sum(dZ) / batch_size
```

For the final softmax layer paired with categorical cross-entropy, the
gradient simplifies to:

```text
dZ = predicted_probabilities - one_hot_true_labels
```

## `MultilayerPerceptron`

| Method | Takes | Returns / changes | Purpose |
| --- | --- | --- | --- |
| `add(layer)` | A built `DenseLayer` | The same model | Adds a layer in forward-pass order. |
| `fit(X, y, epochs, batch_size, learning_rate)` | Prepared training features and labels | The trained model; fills `history` | Shuffles mini-batches, runs forward/backward passes, and updates all layers. |
| `forward(X)` | Feature rows | Softmax probabilities | Sends each batch through every layer. |
| `backward(targets, predictions)` | One-hot labels and output probabilities | Input gradient | Sends categorical-loss error from output to first hidden layer. |
| `predict_proba(X)` | Feature rows | `[P(benign), P(malignant)]` per row | Gives the full softmax result. |
| `predict(X)` | Feature rows | `0` or `1` per row | Chooses the largest softmax probability. |
| `evaluate(X, y)` | Labeled feature rows | Loss and metrics dictionary | Measures a trained model without updating weights. |
| `save(filepath)` | JSON path | Writes model JSON | Saves layers, weights, biases, seed, and scaler. |
| `MultilayerPerceptron.load(filepath)` | Model JSON path | Restored model | Loads a saved model for prediction. |
