# Implementing a Multilayer Perceptron from Scratch

This document explains how to complete `src/model.py` for binary breast-cancer classification. The model receives numerical features, processes them through fully connected layers, and returns the probability that an example belongs to class `1`.

The most natural design for this task is:

```text
X (m, n_features) -> Dense + sigmoid -> ... -> Dense + sigmoid -> p (m, 1)
```

Here `m` is the mini-batch size. Every row is one patient and every column is a feature. The output `p` is a probability, not yet a class label. Convert it to a label using `p >= 0.5`.

## 1. Why an MLP needs activation functions

A dense layer first performs only an affine (linear plus bias) transformation:

```math
Z = XW + b
```

If a network used no activations, then stacking any number of layers would still be equivalent to one affine transformation:

```math
(XW_1+b_1)W_2+b_2 = X(W_1W_2) + (b_1W_2+b_2)
```

So a deep network with no activation function cannot represent curved or complicated decision boundaries. Its hidden layers add parameters but no extra kind of behavior.

An activation introduces non-linearity:

```math
A = g(Z)
```

Now a later layer receives `g(XW+b)` rather than just an affine expression. This is why a network can learn interactions between medical measurements and draw non-linear class boundaries.

Activations must also be differentiable, or differentiable almost everywhere, because training needs their derivatives. Backpropagation uses those derivatives to determine how a small change in a weight changes the loss.

## 2. Sigmoid: meaning and derivative

The sigmoid function maps any real number to `(0, 1)`:

```math
\sigma(z) = \frac{1}{1+e^{-z}}
```

It is useful at the output of a binary classifier because its value can be interpreted as `P(y=1 | x)`. For example, an output of `0.91` means the model assigns a high probability to class `1`.

Its derivative has an especially convenient form. Let `a = sigma(z)`:

```math
\begin{aligned}
\frac{d\sigma}{dz}
&= \frac{d}{dz}(1+e^{-z})^{-1} \\
&= -(1+e^{-z})^{-2}(-e^{-z}) \\
&= \frac{e^{-z}}{(1+e^{-z})^2} \\
&= \frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right) \\
&= \sigma(z)(1-\sigma(z)) = a(1-a).
\end{aligned}
```

Therefore, if the upstream gradient is `dA = dL/dA`, then the chain rule gives:

```math
\frac{dL}{dZ} = \frac{dL}{dA} \odot A(1-A)
```

`odot` means element-wise multiplication. Saving `A` during the forward pass avoids calculating sigmoid a second time during backpropagation.

### Stable sigmoid implementation

For ordinary standardized input ranges, the direct NumPy expression is sufficient:

```python
return 1.0 / (1.0 + np.exp(-Z))
```

Very large negative values can make `np.exp(-Z)` overflow. A robust version computes each branch separately:

```python
out = np.empty_like(Z, dtype=float)
positive = Z >= 0
out[positive] = 1.0 / (1.0 + np.exp(-Z[positive]))
ez = np.exp(Z[~positive])
out[~positive] = ez / (1.0 + ez)
return out
```

## 3. Dense-layer forward pass

For a layer with `n_in` inputs and `n_out` neurons:

| Value | Shape | Meaning |
|---|---:|---|
| `inputs` / `X` | `(m, n_in)` | inputs from the previous layer |
| `W` | `(n_in, n_out)` | learnable connection weights |
| `b` | `(1, n_out)` | one learnable bias per output neuron |
| `Z` | `(m, n_out)` | pre-activation values |
| `A` | `(m, n_out)` | layer outputs |

Implement `DenseLayer.forward` in this order:

```python
self.inputs = inputs
self.Z = inputs @ self.W + self.b       # b broadcasts across the batch
self.A = activation.forward(self.Z)
return self.A
```

The layer must retain `inputs`, `Z`, and `A`: they are required to compute gradients later. This retained data is often called the forward-pass cache.

### Initializing weights

Do not initialize all weights to zero. Every neuron would then produce the same output and receive the same gradient, so the neurons would never learn different features (a symmetry problem).

Initialize small random weights instead. Xavier/Glorot uniform is a good default for sigmoid:

```math
limit = \sqrt{\frac{6}{n_{in}+n_{out}}}, \qquad W \sim U(-limit, limit)
```

```python
self.W = rng.uniform(-limit, limit, size=(input_dim, self.units))
self.b = np.zeros((1, self.units))
```

He initialization is usually better with ReLU, whereas Xavier better preserves the scale of signals for sigmoid or tanh. Standardizing features before training is important for the same reason: it makes optimization more stable and reduces sigmoid saturation.

## 4. Loss: binary cross-entropy (BCE)

For targets `y` in `{0, 1}` and predicted probabilities `p`, the mean BCE loss is:

```math
L = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
```

BCE penalizes confident wrong predictions very strongly. If the true label is `1` but the model outputs almost zero, `log(p)` becomes a large negative number, yielding a large loss.

Before `log`, clip probabilities to prevent `log(0)`:

```python
p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
```

The derivative with respect to the prediction is:

```math
\frac{dL}{dp} = \frac{1}{m}\left(-\frac{y}{p}+\frac{1-y}{1-p}\right).
```

## 5. The key sigmoid + BCE simplification

The output layer has `p = sigma(z)`. Applying the chain rule gives:

```math
\frac{dL}{dz} = \frac{dL}{dp}\frac{dp}{dz}.
```

Substitute the BCE derivative and `dp/dz = p(1-p)`:

```math
\begin{aligned}
\frac{dL}{dz}
&= \left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p) \\
&= -y(1-p) + (1-y)p \\
&= p-y.
\end{aligned}
```

For the *mean* loss, either divide this delta by `m` here, or divide `dW` and `db` by `m` in the dense layer—do it exactly once. A clean convention is:

```python
delta = y_pred - y_true       # dL/dZ for the output layer
```

and average the parameter gradients inside `DenseLayer.backward`.

This simplification is numerically and conceptually useful. Do **not** first call the generic BCE gradient and then multiply by the sigmoid derivative for this paired output/loss case; `p - y` is equivalent and simpler.

## 6. Backpropagation through a dense layer

For `Z = XW+b`, suppose `delta = dL/dZ` has shape `(m, n_out)`. The gradients are:

```math
\begin{aligned}
\frac{dL}{dW} &= \frac{X^T\delta}{m} \\
\frac{dL}{db} &= \frac{\sum_{rows}\delta}{m} \\
\frac{dL}{dX} &= \delta W^T.
\end{aligned}
```

`dX` is the gradient passed to the preceding layer. At a hidden layer, it is not yet `dL/dZ`; it is `dL/dA`. Convert it using that layer's activation derivative:

```math
delta_{hidden} = \frac{dL}{dA_{hidden}} \odot A_{hidden}(1-A_{hidden}).
```

A clear `DenseLayer.backward` interface is:

```python
def backward(self, grad, is_delta=False):
    # is_delta=True means grad already equals dL/dZ.
    delta = grad if is_delta else self.activation.backward(grad, A=self.A)
    m = self.inputs.shape[0]
    self.dW = self.inputs.T @ delta / m
    self.db = np.sum(delta, axis=0, keepdims=True) / m
    return delta @ self.W.T
```

The final layer is called with `is_delta=True` and `grad=y_pred-y_true`. Every hidden layer is called with `is_delta=False`.

### Backpropagation flow

```text
forward:  X -> [Z1, A1] -> [Z2, A2] -> ... -> [ZL, p]
                                           |
                                    deltaL = p - y
                                           |
backward: dW_L, db_L <- deltaL -> dA_(L-1)
                                  -> activation derivative
                                  -> delta_(L-1) -> ... -> dW_1, db_1
```

Important detail: calculate `dX = delta @ W.T` before the optimizer changes `W`, or update weights only after gradients for all layers have been computed.

## 7. Updating parameters with mini-batch SGD

After backpropagation computes gradients, gradient descent takes a small step downhill:

```math
W \leftarrow W - \eta dW, \qquad b \leftarrow b - \eta db
```

where `eta` is the learning rate. The implementation is simply:

```python
layer.W -= self.learning_rate * layer.dW
layer.b -= self.learning_rate * layer.db
```

Training on one example at a time is noisy; training on the entire dataset makes each update slow. Mini-batches balance these choices. For each epoch, shuffle the training rows, split them into batches (for example, 8 rows), and for each batch:

1. run `forward(X_batch)`;
2. compute BCE loss;
3. run `backward(y_batch, prediction)`;
4. update every layer with SGD.

The final batch can be smaller than `batch_size`; the formulas remain valid because they use the actual batch size `m`.

## 8. Wiring the `MultilayerPerceptron`

`add(layer)` should build a layer immediately if a preceding layer already defines its output width. For the first layer, the feature count is known only when `fit` or `forward` first sees `X`.

Suggested lifecycle:

```python
# Build only once, using X.shape[1] for the first layer.
input_dim = X.shape[1]
for layer in self.layers:
    if layer.W is None:
        layer.build(input_dim, rng)
    input_dim = layer.units

# Forward.
A = X
for layer in self.layers:
    A = layer.forward(A)
return A
```

`backward` should start from the output and traverse the list in reverse:

```python
grad = y_pred - y_true
grad = self.layers[-1].backward(grad, is_delta=True)
for layer in reversed(self.layers[:-1]):
    grad = layer.backward(grad, is_delta=False)
```

Ensure target arrays have shape `(m, 1)` so NumPy does not accidentally broadcast `(m,)` against `(m, 1)` into an `(m, m)` matrix:

```python
y = np.asarray(y, dtype=float).reshape(-1, 1)
```

## 9. Sigmoid+BCE versus softmax

The starter file contains both `Sigmoid` and `Softmax`, while the README mentions softmax and binary cross-entropy. These are different valid output conventions, but they must be paired consistently:

| Task representation | Output layer | Target shape | Loss |
|---|---|---|---|
| Binary label `0`/`1` | 1 sigmoid unit | `(m, 1)` | BCE |
| Two one-hot labels, e.g. `[1,0]` | 2 softmax units | `(m, 2)` | categorical cross-entropy |

For the first row, use the `sigmoid + BCE` implementation described above. It fits a CSV target encoded as one binary column.

Softmax converts a vector of class scores into probabilities that sum to one:

```math
softmax(z_j) = \frac{e^{z_j}}{\sum_k e^{z_k}}.
```

For stability, subtract the row maximum before exponentiating; this does not change the probabilities:

```python
shifted = Z - np.max(Z, axis=1, keepdims=True)
exp_z = np.exp(shifted)
return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

With categorical cross-entropy, its output delta also simplifies to `probabilities - one_hot_targets`. Avoid using two softmax outputs with the scalar BCE formula in this project: their assumptions do not match.

## 10. Practical checks before trusting training

1. **Shapes:** print shapes of `W`, `b`, `dW`, and `db` once. They must exactly match their corresponding parameters.
2. **Loss trend:** with a moderate learning rate, training BCE should generally decrease over epochs. It need not decrease every single mini-batch.
3. **Overfit a tiny subset:** train on 8–16 examples until near-perfect accuracy. Failure usually indicates a forward/backward or label-shape bug.
4. **Finite values:** assert `np.isfinite(loss)` and check weights are finite. `NaN` commonly means an unclipped BCE log or an excessively large learning rate.
5. **Gradient check (optional):** perturb one weight by a tiny `epsilon` and compare the numeric slope `(L(w+eps)-L(w-eps))/(2*eps)` with backprop's `dW`. They should be close.

## 11. Common implementation mistakes

- Applying sigmoid only at prediction time instead of in `DenseLayer.forward`.
- Returning a gradient with the wrong orientation. With `X @ W`, the weight gradient is `X.T @ delta`.
- Dividing by the batch size both in output delta and in `dW`/`db`.
- Updating a layer's weights before using those original weights to propagate the gradient backward.
- Letting a one-dimensional target `(m,)` broadcast against predictions `(m, 1)`.
- Fitting a feature scaler on test data. Fit it only on training data; reuse its saved mean and standard deviation for validation and prediction.
- Using zero-initialized weights, which keeps all neurons in a layer identical.
- Using sigmoid in many deep hidden layers without care. Sigmoid can saturate near 0 or 1, where `A(1-A)` is near zero and gradients vanish. For a small two-hidden-layer project it is acceptable; ReLU hidden layers are often preferable in larger networks.

## 12. Minimal architecture for this repository

For 30 standardized input features and two hidden layers of 24 units, construct:

```python
model = MultilayerPerceptron(seed=42)
model.add(DenseLayer(24, activation="sigmoid", initializer="xavier"))
model.add(DenseLayer(24, activation="sigmoid", initializer="xavier"))
model.add(DenseLayer(1, activation="sigmoid", initializer="xavier"))
```

Then train using shuffled mini-batches and evaluate using BCE, accuracy, precision, recall, and a confusion matrix. The saved model must include each layer's units, activation name, `W`, and `b`, plus the training-set scaler parameters so prediction data undergoes the exact same transformation.
