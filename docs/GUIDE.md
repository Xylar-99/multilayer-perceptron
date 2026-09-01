# Multilayer Perceptron — Complete Study Guide

## Project Goal

Build a **from-scratch multilayer perceptron** to classify breast cancer diagnoses as **Malignant (M)** or **Benign (B)** using the Wisconsin Breast Cancer dataset (569 samples, 30 numerical features).

**Constraint:** No neural-network libraries allowed. Only numpy, pandas, and matplotlib for linear algebra and visualization.

---

## 1. Theory You Must Know

### 1.1 What is a Perceptron?
A single neuron computes:
```
weighted_sum = Σ(x_k * w_k) + bias
output = activation(weighted_sum)
```
- `w_k` = weight learned during training
- `bias` = shifts the activation threshold
- `activation` = non-linear function (sigmoid, ReLU, tanh)

### 1.2 What is a Multilayer Perceptron?
- Feedforward network: input → hidden layers → output
- Each layer is **fully connected** (dense)
- At least **2 hidden layers** required for this project
- Data flows only forward; no cycles/recurrent connections

### 1.3 Network Architecture
```
Input Layer (30 features)
    ↓
Hidden Layer 1 (e.g., 24 neurons, sigmoid)
    ↓
Hidden Layer 2 (e.g., 24 neurons, sigmoid)
    ↓
Output Layer (1 neuron, sigmoid) → probability [0, 1]
```

### 1.4 Activation Functions

| Function | Formula | When to Use |
|----------|---------|-------------|
| **Sigmoid** | `σ(x) = 1 / (1 + e⁻ˣ)` | Output layer for binary classification |
| **ReLU** | `max(0, x)` | Hidden layers (avoids vanishing gradients) |
| **Tanh** | `tanh(x)` | Hidden layers (zero-centered) |
| **Softmax** | `eˣⁱ / Σeˣʲ` | Output layer for multi-class classification |

### 1.5 Weight Initialization
- **He Uniform**: `W ~ Uniform(-√(6/n), √(6/n))` — use with ReLU
- **Xavier Uniform**: `W ~ Uniform(-√(6/(n+m)), √(6/(n+m)))` — use with sigmoid/tanh

### 1.6 Loss Function — Binary Cross-Entropy
```
Loss = -[y * log(p) + (1-y) * log(1-p)]
```
- `y` = true label (0 or 1)
- `p` = predicted probability
- Measures how far predictions are from true labels

### 1.7 Forward Pass (Feedforward)
For each layer `l`:
```
Z_l = A_{l-1} @ W_l + b_l
A_l = activation(Z_l)
```
- `Z` = pre-activation (linear combination)
- `A` = post-activation (non-linear output)

### 1.8 Backpropagation
Compute gradients from output to input using chain rule:

**Output layer:**
```
dA = derivative of loss w.r.t. prediction
dZ = dA * activation_derivative(Z)
dW = (A_prev.T @ dZ) / m
db = mean(dZ, axis=0)
dA_prev = dZ @ W.T
```

**Hidden layers:** Same formula, backpropagated through each layer

**Key insight:** Gradient tells you how much each weight contributed to the error.

### 1.9 Gradient Descent
Update weights to minimize loss:
```
W = W - learning_rate * dW
b = b - learning_rate * db
```

### 1.10 Mini-Batch Gradient Descent
Instead of using all samples at once:
1. Split data into mini-batches (e.g., 8 samples)
2. For each batch: forward → loss → backward → update
3. After all batches: 1 epoch complete
4. Repeat for N epochs

**Benefits:**
- Faster convergence than full-batch
- Noisy updates help escape local minima
- Memory efficient for large datasets

### 1.11 Learning Curves
Track during training:
- **Training loss/accuracy**: Model performance on training data
- **Validation loss/accuracy**: Model performance on unseen data

**Interpretation:**
- Both decrease → learning well
- Training ↓, Validation ↑ → overfitting
- Both flat → underfitting or learning rate too small

---

## 2. Project Parts

### Part 1: Split Data
**Goal:** Separate raw data into train and validation sets.

**Steps:**
1. Load `data/data.csv` (569 rows, 32 columns)
2. Shuffle with `np.random.RandomState(seed=42)` for reproducibility
3. Split: 80% train, 20% validation
4. Save to `data/train.csv` and `data/test.csv`

**Important:** Split BEFORE any training. Never let test data influence training.

### Part 2: Train Model
**Goal:** Build and train a neural network from scratch.

**Steps:**
1. Load `train.csv` and `test.csv`
2. Normalize features:
   - Compute mean/std from **training data only**
   - Apply same normalization to validation
3. Build network:
   ```python
   model = Model()
   model.add(Dense(24, activation='sigmoid'))
   model.add(Dense(24, activation='sigmoid'))
   model.add(Dense(1, activation='sigmoid'))
   ```
4. Train loop:
   - For each epoch:
     - Shuffle training data
     - For each batch: forward → loss → backward → update
     - Compute metrics on train and validation
   - Save model to `saved_model.npy`
   - Plot learning curves

**Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layer` | 24 24 | Hidden layer sizes |
| `--epochs` | 84 | Number of training iterations |
| `--learning_rate` | 0.0314 | Step size for gradient descent |
| `--batch_size` | 8 | Samples per gradient update |
| `--activation` | sigmoid | Hidden layer activation |
| `--optimizer` | sgd | sgd / sgd_momentum / adam / rmsprop |
| `--loss` | binaryCrossentropy | Loss function |
| `--seed` | 42 | Random seed for reproducibility |

### Part 3: Predict
**Goal:** Load saved model and evaluate on new data.

**Steps:**
1. Load model from `saved_model.npy`
2. Load and normalize input data using saved mean/std
3. Forward pass through network
4. Convert probabilities to classes (threshold 0.5)
5. Calculate and display accuracy

---

## 3. Implementation Roadmap

### Step 1: Utility Functions (`src/utils.py`)
```python
def normalize(X):          # Standardize to zero mean, unit variance
def shuffle_data(X, y):    # Shuffle with reproducibility
def one_hot(y, classes):   # Convert labels to one-hot vectors
```

### Step 2: Activation Functions (`src/layers.py`)
Implement `forward` and `backward` for:
- `sigmoid`
- `relu`
- `tanh`
- `softmax`

### Step 3: Dense Layer (`src/layers.py`)
```python
class Dense:
    def __init__(self, units, activation, initializer)
    def forward(X):        # Z = X @ W + b, A = activation(Z)
    def backward(dA):      # Compute dW, db, return dX
    def update(optimizer): # Update W and b
```

### Step 4: Loss Functions (`src/losses.py`)
```python
class BinaryCrossEntropy:
    def compute(y_true, y_pred)
    def derivative(y_true, y_pred)
```

### Step 5: Optimizers (`src/optimizers.py`)
```python
class SGD:
    def update(W, b, dW, db)
class Adam:
    def update(W, b, dW, db)  # Maintains momentum and adaptive learning rates
```

### Step 6: Model (`src/model.py`)
```python
class Model:
    def add(layer)
    def forward(X)           # Propagate through all layers
    def backward(y, y_pred)  # Backpropagate error
    def update(optimizer)    # Update all weights
    def predict(X)           # Forward pass for inference
```

### Step 7: Scripts
- `src/split.py` — Data splitting
- `src/train.py` — Full training loop
- `src/predict.py` — Model evaluation
- `mlp.py` — CLI entry point

---

## 4. Common Pitfalls to Avoid

1. **Data leakage:** Don't normalize validation data with global statistics. Fit normalization on train only.
2. **Wrong gradient shapes:** Matrix dimensions must match at every backprop step.
3. **Exploding/vanishing gradients:** Use proper initialization (He/Xavier) and consider ReLU.
4. **Forgetting to zero gradients:** Ensure `dW` and `db` are computed fresh each batch.
5. **Softmax stability:** Subtract max before exponentiation to avoid overflow.
6. **Learning rate too high/low:** Start with 0.01–0.05 and adjust based on loss curves.

---

## 5. Evaluation Criteria

- **Understanding:** You will explain feedforward, backpropagation, and gradient descent to a corrector.
- **Correctness:** Model trains, saves, loads, and predicts correctly.
- **Accuracy:** Validation accuracy should exceed ~95% on this dataset.
- **Code quality:** Clear structure, no ML libraries, well-commented.

---

## 6. Recommended Videos

### Neural Networks Fundamentals
- [But what is a neural network? — 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk)
- [Gradient descent, how neural networks learn — 3Blue1Brown](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [What is backpropagation really doing? — 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### Implementation Tutorials
- [Neural Network from scratch in Python — Coding Train](https://www.youtube.com/watch?v=aircAruvnKk)
- [Building a neural network from scratch in Python — Stats Quest](https://www.youtube.com/watch?v=w8yWXqWQYmI)
- [Python neural network backpropagation — sentdex](https://www.youtube.com/watch?v=aircAruvnKk)

### Advanced Topics (Bonuses)
- [Adam optimizer explained — statquest](https://www.youtube.com/watch?v=kyXf97X3W4U)
- [Batch normalization — 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk)
- [Regularization and dropout — Andrew Ng](https://www.youtube.com/watch?v=aircAruvnKk)

---

## 7. Quick Reference Commands

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Split data
python -m src.split --dataset data/data.csv --seed 42

# Train
python -m src.train --dataset data/data.csv --layer 24 24 --epochs 84 --learning_rate 0.0314 --batch_size 8

# Predict
python -m src.predict --dataset data/test.csv --model saved_model.npy
```

---

## 8. File Structure

```
multilayer-perceptron/
├── data/
│   └── data.csv
├── docs/
│   └── GUIDE.md
├── en.subject.pdf
├── .gitignore
├── mlp.py
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── layers.py
    ├── model.py
    ├── losses.py
    ├── optimizers.py
    ├── utils.py
    ├── metrics.py
    ├── callbacks.py
    ├── visualization.py
    ├── split.py
    ├── train.py
    └── predict.py
```
