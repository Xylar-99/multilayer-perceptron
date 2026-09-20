# Multilayer Perceptron — Breast Cancer Classification

This project implements a binary Multilayer Perceptron (MLP) from scratch
with NumPy. It classifies the Wisconsin Breast Cancer labels:

- `M` (malignant) becomes `1`.
- `B` (benign) becomes `0`.

The project is organized into split, train, and predict modules around a
NumPy implementation of dense layers, ReLU, softmax, and cross-entropy.

## Project flow

```text
raw CSV
  -> --split cleans, splits, Min-Max scales, and saves the datasets
  -> --train trains the MLP, saves learning curves, and saves the model
  -> --predict loads the model and an external scaler, then writes predictions
```

The default network has this shape:

```text
input features -> ReLU hidden layer(s) -> 2-unit softmax output
```

Softmax returns two probabilities: `[P(benign), P(malignant)]`. They sum to
`1`, and the larger one chooses class `0` (benign) or class `1` (malignant).

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the normal workflow

Run these commands from the project root.

```bash
# 1. Split, Min-Max scale, and save the data.
python mlp.py --split --dataset data/data.csv

# 2. Train a model using the generated files.
python mlp.py --train --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314

# 3. Evaluate the saved model on the test set.
python mlp.py --predict --dataset data/test.csv --model output/saved_model.json --scaler output/scaler.json
```

The split command creates `data/train.csv`, `data/test.csv`, and
`output/scaler.json`. The generated train and test CSV files contain a
generated numeric ID, a `B`/`M` diagnosis, and 30 scaled features.

Training saves the network topology and learned weights and biases to
`output/saved_model.json`. The scaler is saved separately in
`output/scaler.json`; it is not stored in the trained model JSON.

At the end of training, the program saves loss and accuracy learning curves to
`output/learning_curves.png` and displays the figure.

### Use custom output paths

The split outputs, model path, plot path, and prediction output path can be
changed with their corresponding options. Pass the scaler created during
splitting to prediction:

```bash
python mlp.py --split \
  --train_out prepared/train.csv \
  --test_out prepared/test.csv \
  --scaler_out prepared/scaler.json

python mlp.py --train \
  --train_data prepared/train.csv \
  --test_data prepared/test.csv \
  --model_out prepared/model.json \
  --plot_out prepared/learning_curves.png

python mlp.py --predict \
  --dataset prepared/test.csv \
  --model prepared/model.json \
  --scaler prepared/scaler.json \
  --output_csv prepared/predictions.csv
```

## What each file does

| File | Responsibility |
| --- | --- |
| `mlp.py` | Parses command-line options and starts split, train, or predict mode. |
| `src/data.py` | Defines `Dataset` plus CSV, scaling, and scaler-saving helpers. |
| `src/split.py` | Contains the straightforward clean → split → scale → save workflow. |
| `src/model.py` | Contains ReLU, softmax, categorical loss, dense layers, training, evaluation, and model persistence. |
| `src/train.py` | Builds the ReLU-plus-softmax network, records training and validation metrics, plots, and saves the model. |
| `src/predict.py` | Loads a model and external scaler, then predicts supported CSV formats. |

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

## Prediction CSV formats

The predictor accepts either of these headerless formats:

- `ID, diagnosis, feature_1, ..., feature_30`: the diagnosis must be `B` or
  `M`. The predictor treats these features as already scaled, writes
  predictions, and prints binary cross-entropy. The generated `data/test.csv`
  has this format.
- `ID, feature_1, ..., feature_30`: the predictor applies the supplied scaler,
  then writes predictions. Use unscaled features in this format.

A feature-only file with exactly 30 columns and no leading ID is not accepted.
Prediction output is a CSV with `id` and `predict` columns; its default path is
`data/predictions.csv`.
