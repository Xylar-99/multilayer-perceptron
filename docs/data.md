# `src/data.py` — dataset and scaling helpers

## Goal

This file turns CSV rows into a `Dataset`, removes unusable rows, creates the
Min-Max scaler, and saves or loads scaler settings. `Dataset` remains a class
because it owns two related pieces of state: feature values (`X`) and labels
(`y`).

## `Dataset`

| Method | Takes | Returns / changes | Why it helps |
| --- | --- | --- | --- |
| `Dataset(X, y)` | Feature matrix and label array | A dataset object | Keeps matching features and labels together. |
| `Dataset.from_csv(filepath, allow_missing_diagnosis=False)` | Headerless CSV path | A labeled `Dataset`, or `None` when missing labels are allowed | Reads the raw or prepared project CSV format. |
| `Dataset.from_frame(raw_data, allow_missing_diagnosis=False)` | Pandas DataFrame | A labeled `Dataset`, or `None` | Lets prediction inspect a CSV once instead of reading it twice. |
| `clean()` | No new input | The same dataset with incomplete rows removed | Converts feature values to numbers, then removes missing/non-numeric rows. |
| `split(train_ratio, rng=None)` | Fraction for training and optional random generator | `(train_data, test_data)` | Shuffles matching `X`/`y` rows and separates them reproducibly when a seed is supplied. |
| `fit_scaler()` | No new input | Dictionary with feature minimums and maximums | Learns scaling values from training data only. |
| `scale(scaler)` | Scaler dictionary | The same dataset with scaled `X` | Applies the training scaler without changing labels. |
| `save_csv(filepath)` | Destination path | Writes headerless feature-plus-label CSV | Saves prepared train or test data. |

## Module functions

| Function | Takes | Returns / does | Why it helps |
| --- | --- | --- | --- |
| `save_scaler(scaler, filepath)` | Scaler dictionary and JSON path | Writes JSON | Makes the exact training scaling reusable later. |
| `load_scaler(filepath)` | Scaler JSON path | Scaler dictionary | Restores saved minimums and maximums. |
| `scale_features(features, scaler)` | Numeric feature matrix and scaler | New scaled matrix | Pure scaling math used when a full `Dataset` object is unnecessary. |
| `features_need_scaling(features)` | Feature matrix | `True` for raw Wisconsin measurements | Prevents prepared train/test CSVs from being scaled twice. |

Min-Max scaling uses:

```text
scaled_value = (value - training_minimum) / (training_maximum - training_minimum)
```

For a constant feature, the code uses a denominator of `1` so it never divides
by zero.
