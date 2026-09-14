# Multilayer Perceptron — Breast Cancer Classification

This project implements a binary Multilayer Perceptron (MLP) from scratch
with NumPy. It classifies the Wisconsin Breast Cancer labels:

- `M` (malignant) becomes `1`.
- `B` (benign) becomes `0`.

The code is deliberately organized so its three stateful classes hold real
data or model state, while the split, train, and predict workflows stay as
simple module-level functions.

## Project flow

```text
raw CSV
  -> split_dataset loads, cleans, splits, and scales the rows
  -> train_model builds, trains, evaluates, plots, and saves the MLP
  -> predict_from_file loads the model and evaluates or predicts
```

## Learn each file

For an explanation of every core class and function—its goal, inputs, result,
and role in the workflow—start with the [file-by-file code guide](docs/README.md).

The network created by `create_network` has this shape:

```text
input features -> ReLU hidden layer(s) -> 2-unit softmax output
```

Softmax returns two probabilities: `[P(benign), P(malignant)]`. They sum to
`1`, and the larger one chooses class `0` (benign) or class `1` (malignant).

## Visual guide: two hidden layers

The command `--layer 24 24` builds the example below: the 30 scaled breast
cancer measurements enter two fully connected ReLU layers, each with 24
neurons. The final two softmax neurons return the probabilities for benign and
malignant diagnoses.

![Diagram of a 30-input MLP with two 24-neuron hidden layers](images/two-hidden-layer-network.svg)

Every neuron takes all values from the preceding layer, gives each connection
a learned weight, adds a learned bias, and applies its activation function.
The lines in the diagram are representative: in the real network, every
neuron is connected to every neuron in the next layer.

## Visual guide: how training works

Training takes a small batch of patient rows, makes softmax predictions,
measures categorical error, and sends that error backward to improve each
weight and bias.

![Diagram of the MLP mini-batch training cycle](images/training-cycle.svg)

For example, this trains the illustrated network for 84 passes over the
training data, using batches of eight rows:

```bash
python mlp.py --train --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
```

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the normal workflow

Run these commands from the project root.

```bash
# 1. Shuffle, split, Min-Max scale, and save the data.
python mlp.py --split --dataset data/data.csv --seed 42

# 2. Train a model using the generated files.
python mlp.py --train --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314

# 3. Evaluate the saved model on the test set.
python mlp.py --predict --dataset data/test.csv --model output/saved_model.json
```

The split command creates `data/train.csv`, `data/test.csv`, and
`output/scaler.json`. Training saves the scaler inside the model JSON, along
with its layers, weights, and biases.

### Use custom output paths

If the scaler is not saved at the default location, pass its path when
training so the exact train-set scaler is stored with the model:

```bash
python mlp.py --split \
  --train_out prepared/train.csv \
  --test_out prepared/test.csv \
  --scaler_out prepared/scaler.json \
  --seed 42

python mlp.py --train \
  --train_data prepared/train.csv \
  --test_data prepared/test.csv \
  --scaler prepared/scaler.json \
  --model_out prepared/model.json
```

`--scaler` is also available with `--predict` when an older model does not
contain a scaler or when you deliberately need to override the saved one.

## What each file does

| File | Responsibility |
| --- | --- |
| `mlp.py` | Parses command-line options and starts split, train, or predict mode. |
| `src/data.py` | Defines `Dataset` plus CSV, scaling, and scaler-saving helpers. |
| `src/split.py` | Contains the straightforward clean → split → scale → save workflow. |
| `src/model.py` | Contains ReLU, softmax, categorical loss, dense layers, training, evaluation, and model persistence. |
| `src/train.py` | Builds the ReLU-plus-softmax network and runs training, test evaluation, plotting, and saving. |
| `src/predict.py` | Evaluates labeled data or predicts feature-only rows with a saved model. |

## The important training math

For each dense layer, the forward pass is:

```text
Z = inputs @ weights + biases
A = activation(Z)
```

For this two-class classifier, the final activation is softmax and the loss is
categorical cross-entropy. The true label is one-hot encoded: benign is
`[1, 0]` and malignant is `[0, 1]`. Their combined output-layer gradient is:

```text
dZ = predicted_probabilities - one_hot_true_labels
```

Each layer then calculates:

```text
dW = inputs.T @ dZ / batch_size
db = sum(dZ) / batch_size
dA_previous = dZ @ weights.T
```

Training shuffles the rows, works through mini-batches, updates each layer,
and records loss and accuracy after every epoch.

## Use the model in Python

```python
from src.train import create_network

model = create_network([24, 24], input_features=30, seed=42)
```

For normal use, `create_network([24, 24], input_features=30)` builds this
pattern for you. `model.predict_proba(X)` then returns one row per sample with
`[P(benign), P(malignant)]`.

## CSV formats

The raw Wisconsin dataset contains an ID column, an `M`/`B` diagnosis column,
and 30 numeric features. The generated train and test files contain the 30
scaled features followed by the diagnosis column.

Prediction also supports a feature-only CSV with 30 numeric columns. A file
with one leading ID column is accepted as well.
