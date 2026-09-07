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
