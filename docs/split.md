# `src/split.py` — prepare train/test data

## Goal

`split.py` contains one workflow function, `split_dataset(...)`. It prepares
the raw Wisconsin CSV for training while preventing information from the test
set from leaking into the scaler.

## `split_dataset`

```python
split_dataset(dataset_path, train_out, test_out, scaler_out, ratio=0.8, seed=None)
```

| Input | Meaning |
| --- | --- |
| `dataset_path` | Raw CSV with an ID column, `M`/`B` diagnosis column, and 30 features. |
| `train_out` | Where to save the prepared training CSV. |
| `test_out` | Where to save the prepared test CSV. |
| `scaler_out` | Where to save the training-only Min-Max scaler JSON. |
| `ratio` | Fraction of rows placed in training, normally `0.8`. |
| `seed` | Optional integer that makes the shuffle repeatable. |

## Steps and result

1. Load and clean the raw data.
2. Shuffle and split rows into train and test datasets.
3. Fit the scaler on the train features only.
4. Scale both train and test features with that same scaler.
5. Save the two CSVs and the scaler JSON.

It returns `(train_data, test_data, scaler)` as a convenience for Python use.
The CLI usually relies on the saved files instead.
