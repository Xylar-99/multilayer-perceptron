# Multilayer Perceptron — Breast Cancer Classification

A clean, from-scratch implementation of a **Multilayer Perceptron (MLP)** for the Wisconsin Breast Cancer dataset, built using only pure Python, NumPy, and Matplotlib (no ML libraries allowed).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full modular design breakdown, mathematical derivations, and defense preparation.

---

## Quick Start

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Single Program Mode (`mlp.py`)
```bash
# 1. Split dataset into train (80%) and validation/test (20%)
python mlp.py --split --dataset data/data.csv --seed 42

# 2. Train network with at least 2 hidden layers (default: 24 24)
python mlp.py --train --layer 24 24 --epochs 84 --learning_rate 0.0314 --batch_size 8

# 3. Predict & evaluate using Binary Cross-Entropy Loss
python mlp.py --predict --dataset data/test.csv --model output/saved_model.json
```

### 3. Modular Scripts Mode (`src/`)
```bash
# Split
python -m src.split --dataset data/data.csv --seed 42

# Train
python -m src.train --dataset data/train.csv --val_dataset data/test.csv --layer 24 24 --epochs 84

# Predict
python -m src.predict --dataset data/test.csv --model output/saved_model.json
```

### Create a network in Python

`DenseLayer` describes one fully connected layer. The final layer has one
sigmoid unit because this project predicts a binary label (`M` or `B`).

```python
from src.model import DenseLayer, MultilayerPerceptron

model = MultilayerPerceptron(seed=42)
model.add(DenseLayer(24, activation="sigmoid"))  # hidden layer 1
model.add(DenseLayer(24, activation="sigmoid"))  # hidden layer 2
model.add(DenseLayer(1, activation="sigmoid"))   # binary output
```

The trainer creates that same pattern automatically from `--layer 24 24`:

```python
from src.train import ModelTrainer

model = ModelTrainer.create_network([24, 24], seed=42)
```

---

## Clean & Compact Structure

```
multilayer-perceptron/
├── data/
│   ├── data.csv            # Original Wisconsin Diagnostic Breast Cancer dataset
│   ├── train.csv           # Generated train split (80%)
│   └── test.csv            # Generated validation/test split (20%)
├── output/
│   ├── saved_model.json    # Human-readable saved model (weights, biases, topology, scaler)
│   └── learning_curves.png # Saved loss and accuracy learning curve graphs
├── docs/
│   └── ARCHITECTURE.md     # In-depth architectural & mathematical documentation
├── src/
│   ├── __init__.py
│   ├── data.py             # Dataset loader, mini-batch generator & StandardScaler
│   ├── model.py            # DenseLayer, Activations, Losses, Optimizers & MultilayerPerceptron
│   ├── split.py            # Dataset split script
│   ├── train.py            # Model training script
│   └── predict.py          # Prediction & evaluation script
├── mlp.py                  # Single-script CLI entry point
├── requirements.txt        # Dependencies (numpy, matplotlib)
└── README.md
```

---

## Features & Subject Compliance

| Subject Requirement | Implementation |
|---|---|
| **No ML Libraries** | Coded from scratch in NumPy / Python standard library |
| **Data Separation** | `src/split.py` / `mlp.py --split` with seed reproducibility & no data leakage |
| **Standardization** | Z-score scaler fitted strictly on train data and stored in model JSON |
| **Modular Topology** | `--layer 24 24 24` allows arbitrary hidden layer architectures |
| **Softmax Output** | Softmax activation on output layer yielding probabilistic distributions |
| **Learning Curves** | Epoch metrics logged + plots saved to `output/learning_curves.png` |
| **Model Persistence** | Saved in human-readable JSON (with full support for `.npy` format as well) |
| **Binary Cross-Entropy** | Prediction evaluated with exact subject formula $E = -\frac{1}{N}\sum [y\log p + (1-y)\log(1-p)]$ |
| **Bonus Optimizers** | Mini-batch SGD, SGD with Momentum, RMSprop, and Adam |

## Simple Training Flow

Run the commands in this order:

```bash
python3 mlp.py --split --dataset data/data.csv
python3 mlp.py --train --epochs 84 --batch_size 8 --learning_rate 0.01
python3 mlp.py --predict
```

The split step saves the feature scaler. Training saves the same scaler inside
the model file, so prediction can also work with the original unscaled CSV.

The network uses this simple structure:

```text
features -> ReLU hidden layers -> one sigmoid output
```

The sigmoid output is a probability. A probability of `0.5` or higher becomes
class `1`; otherwise it becomes class `0`. In this dataset, `M` is converted to
`1` and `B` is converted to `0`.

During training, the model shuffles the rows, trains one mini-batch at a time,
updates the weights, and records loss and accuracy. The training command saves
the learning plot to `output/learning_curves.png`.

Prediction accepts two kinds of CSV files:

- A labeled file with 30 feature columns and an `M`/`B` column. It returns
  loss, accuracy, precision, recall, and a confusion matrix.
- A feature-only file with 30 numeric columns. It returns one probability and
  one class for every row. You can also include an ID as the first column.

For a feature-only file, class `1` means `M` and class `0` means `B`:

```text
Row 1: class=1, probability=0.9321
Row 2: class=0, probability=0.1045
```

The sigmoid implementation uses separate positive and negative calculations.
This avoids `exp()` overflow when a value is very large or very small. Weights
are also initialized with small values, which prevents the network from
starting with saturated sigmoid outputs and predicting only one class.
