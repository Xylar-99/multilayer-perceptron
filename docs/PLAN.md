# Multilayer Perceptron — 12-Day Plan

Goal: complete and deeply understand the mandatory part, then add safe bonuses.

Schedule: **12 focused days × 4 hours = 48 hours**.

## Daily study method

Use each four-hour session like this:

- **60–90 minutes:** theory, formulas, and a small calculation by hand
- **120–150 minutes:** implementation
- **30–45 minutes:** tests, notes, Git commit, and verbal explanation

Do not keep a formula in the project unless you can explain:

1. What every symbol and matrix shape means
2. Where the formula comes from
3. Why the project needs it
4. What happens numerically when it is wrong

## Day 1 — Dataset and project requirements

### Theory

- Read the complete subject and convert every mandatory requirement into a checklist.
- Understand features, labels, samples, classes, training data, and validation data.
- Study mean, variance, standard deviation, and z-score normalization:

  `x_normalized = (x - mean) / standard_deviation`

- Understand why the mean and standard deviation must come only from training data.

### Implementation

- Explore the CSV: shape, missing values, class counts, and feature ranges.
- Design the project structure and command-line interfaces.
- Write small data-loading tests.

### Explain at the end

- Why the ID column is not a feature
- Why unscaled features make optimization difficult
- Why validation data must not influence training

## Day 2 — Split and preprocessing program

### Theory

- Study random and stratified train/validation splitting.
- Understand reproducibility and random seeds.
- Understand data leakage.

### Implementation

- Implement the required dataset-separation program.
- Preserve approximately the same class distribution in both sets.
- Add input validation and reproducible seed handling.
- Test row counts, duplicates, overlap, and class distribution.

### Deliverable

- The first complete mandatory program

## Day 3 — Neurons, dense layers, and matrix shapes

### Theory

- Derive a neuron:

  `z = Σ(x_i * w_i) + b`

  `a = activation(z)`

- Generalize it to a batch and dense layer:

  `Z = XW + b`

- Trace all shapes for a network such as `30 → 24 → 24 → 2`.
- Understand why the bias exists and why nonlinear activations are necessary.
- Study sigmoid, tanh, and ReLU, including their derivatives.

### Implementation

- Implement activation functions and derivatives.
- Implement a dense layer and feedforward pass.
- Add shape and numerical tests.

### By-hand exercise

- Compute the output of a tiny `2 → 2` layer using actual numbers.

## Day 4 — Softmax, cross-entropy, and predictions

### Theory

- Derive stable softmax:

  `softmax(z_i) = exp(z_i - max(z)) / Σ exp(z_j - max(z))`

- Understand why subtracting the maximum does not change the result.
- Study one-hot encoding and categorical cross-entropy:

  `L = -(1/N) Σ_n Σ_c y_nc log(p_nc)`

- Understand binary versus categorical cross-entropy.
- Derive why softmax outputs form a probability distribution.

### Implementation

- Implement stable softmax, cross-entropy, prediction, and accuracy.
- Test extreme input values and protect against `log(0)`.

### By-hand exercise

- Calculate softmax and cross-entropy for two classes manually.

## Day 5 — Backpropagation derivation

### Theory

- Review derivatives, partial derivatives, and the chain rule.
- Derive, without copying, the output error:

  `dZ_output = predictions - targets`

- Derive dense-layer gradients:

  `dW = A_previous.T @ dZ / batch_size`

  `db = sum(dZ) / batch_size`

  `dA_previous = dZ @ W.T`

- Derive hidden-layer error:

  `dZ = dA * activation_derivative(Z)`

- Understand why gradients flow backward and why they are averaged over a batch.

### Implementation

- Implement backward propagation one layer at a time.
- Add a numerical gradient check using finite differences.

### Defense target

- Explain every multiplication and matrix shape on a whiteboard.

## Day 6 — Gradient descent and training loop

### Theory

- Derive the update:

  `parameter = parameter - learning_rate * gradient`

- Understand full-batch, stochastic, and mini-batch gradient descent.
- Study epochs, batch size, learning rate, shuffling, underfitting, and overfitting.
- Understand Xavier/Glorot and He initialization and why zero initialization fails.

### Implementation

- Implement initialization, mini-batches, shuffling, training, and validation.
- Print training and validation loss and accuracy each epoch.
- Confirm that loss decreases on a tiny dataset before training on the full dataset.

## Day 7 — Modular network and model persistence

### Theory

- Understand why prediction must use exactly the same architecture and normalization.
- Decide what a saved model must contain:
  architecture, activation names, weights, biases, class mapping, mean, and standard deviation.

### Implementation

- Support at least two hidden layers by default.
- Make layer sizes and training parameters configurable.
- Save and load the complete model safely.
- Test that predictions are identical before and after reloading.

## Day 8 — Prediction program and mandatory graphs

### Theory

- Study confusion matrices, accuracy, precision, recall, and F1 score.
- Understand why cancer false negatives are especially important.
- Learn to interpret training and validation curves.

### Implementation

- Complete the required prediction program.
- Report the required binary cross-entropy evaluation.
- Generate both mandatory graphs:
  training/validation loss and training/validation accuracy.
- Add clear errors for malformed files and incompatible models.

## Day 9 — Mandatory verification and debugging

### Verification checklist

- Run the whole workflow from a clean environment.
- Split data, train, save, load, predict, and display graphs.
- Test different seeds, layer sizes, batch sizes, and learning rates.
- Check for `NaN`, overflow, zero standard deviation, and incorrect shapes.
- Compare selected calculations with a trusted manual calculation.
- Review the subject line by line.

### Rule

Do not begin bonus work until every mandatory requirement works reliably.

## Day 10 — First bonus: early stopping and training history

### Theory

- Understand patience, minimum improvement, and restoring the best weights.
- Understand how early stopping reduces overfitting.

### Implementation

- Add early stopping with best-model restoration.
- Save metric history.
- Optionally plot multiple experiments for comparison.
- Keep bonus behavior optional so it cannot break mandatory behavior.

## Day 11 — Second bonus: Adam or additional metrics

### Theory

- First study momentum and RMSprop.
- Then derive Adam's first and second moment estimates, bias correction, and update.
- Be prepared to explain every new state variable and hyperparameter.

### Implementation

- Prefer one well-tested bonus:
  Adam optimizer **or** precision/recall/F1 and confusion matrix.
- Compare the bonus objectively against standard gradient descent.
- Do not include an optimizer you cannot derive and defend.

## Day 12 — Final audit, documentation, and mock defense

### Implementation and submission

- Run the complete workflow again.
- Clean the repository without deleting useful work.
- Write usage examples and dependency/setup instructions.
- Confirm that only repository contents are required for evaluation.
- Review Git status and push the final tested commits.

### Mock defense

Explain without reading:

1. One complete feedforward pass, with shapes
2. Softmax and cross-entropy, including numerical stability
3. The chain rule and every backpropagation equation
4. Gradient descent and the effects of its hyperparameters
5. Initialization, normalization, splitting, and data leakage
6. Training versus validation curves and overfitting
7. Saving/loading and consistent prediction preprocessing
8. Every implemented bonus and why it works

## Definition of done

The project is ready only when:

- All three mandatory phases work from the command line.
- The default network has at least two hidden layers and a softmax output.
- Backpropagation and gradient descent are implemented from scratch.
- Training and validation loss and accuracy are displayed per epoch.
- Both required learning-curve graphs are generated.
- The complete model and preprocessing statistics are saved and restored.
- Prediction reports the required loss.
- Invalid inputs fail with understandable messages.
- The implementation has repeatable tests.
- You can derive and explain every important formula.
- Mandatory functionality remains correct when bonuses are disabled.

## Important expectation

Forty-eight focused hours is a strong target for a first complete version and solid
understanding. Deep understanding comes from derivation, implementation, debugging,
and explanation—not from finishing many bonuses. If mandatory debugging takes longer,
drop a bonus rather than weakening the core project or defense preparation.
