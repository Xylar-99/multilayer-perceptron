# `src/predict.py` — evaluate or predict with a saved model

## Goal

This file loads a saved model and uses it in one of two ways:

1. A CSV with `M`/`B` labels is evaluated and receives loss/accuracy metrics.
2. A feature-only CSV receives one softmax prediction per row.

## Functions

| Function | Takes | Returns / does | Why it helps |
| --- | --- | --- | --- |
| `print_evaluation(results, title='Test Set Evaluation')` | Evaluation dictionary | Prints a formatted report | Gives train and predict workflows one consistent metric display. |
| `scale_if_needed(features, scaler)` | Feature matrix and optional scaler | Prepared feature matrix | Scales raw Wisconsin values while leaving prepared CSV values alone. |
| `predict_from_file(dataset_path, model_path, scaler_path=None)` | Data CSV, saved-model path, optional scaler override | Metrics dictionary or probability/prediction dictionary | Loads once, decides whether labels are present, then evaluates or predicts. |

## Feature-only prediction result

For each row, `predict_from_file` returns and prints:

```text
P(benign)
P(malignant)
predicted class: 0 (benign) or 1 (malignant)
```

The class is selected with `argmax`: whichever softmax output probability is
larger wins. A feature-only CSV must have 30 numeric columns. One leading ID
column is also accepted and removed automatically.
