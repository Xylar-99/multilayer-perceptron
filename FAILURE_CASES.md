# Failure Cases and Model-File Checklist

This is a documentation-only audit of the current project behavior. It does
not change the implementation. Use it to understand why a command fails, what
the model JSON must contain, and which inputs are unsafe.

## Quick answer for model files

| Item in the model JSON | Required for loading? | Current behavior |
| --- | --- | --- |
| `input_features` | Yes | Missing it fails with `Error: 'input_features'`. This number is used to check prediction CSV width. |
| `hidden_layers` | Yes | Missing it fails with `Error: 'hidden_layers'`. |
| `layers` | Yes | Missing it fails with `Error: 'layers'`. Each saved layer also needs numeric `weights` and `biases`. |
| `seed` | No | Missing it falls back to `42`; it is not necessary to make a prediction. |
| `scaler` | No in the JSON | The current saved model has `"scaler": null`. Prediction still requires the separate scaler file passed with `--scaler`. |
| `learning_rate` | No | It is not stored and is not needed for prediction. It is only a training setting. |
| `epochs`, `batch_size`, history | No | They are not stored. Keep the original training command if exact training settings must be retained. |

An edited, incomplete, or incompatible model JSON is not validated before
inference. Keep the model JSON and its matching scaler JSON together and do
not hand-edit either file.

## Critical known model-load limitation

The current save/load format is not architecture-faithful:

1. `save()` stores `hidden_layers` including the final output layer `2`.
2. `load()` passes that list to the constructor.
3. The constructor appends another final `2` layer.

For example, the current saved model contains three layers and
`hidden_layers = [24, 24, 2]`, but it loads as four layers:

```text
saved JSON: [24, 24, 2]
loaded model: [24, 24, 2, 2]
```

The saved output layer is then treated as a ReLU layer and an extra random
softmax layer remains. There is also a seed-loading issue: the saved seed is
passed as `epochs` rather than as `seed`, so a non-default saved seed is not
restored. The current example happens to run, but a loaded model should be
treated as structurally unverified until this is fixed.

Sources: [src/model.py](src/model.py) lines 127-146 and 231-269.

## Prediction input contract

The predictor accepts **raw, unscaled numeric features** only. It always
applies the scaler before inference.

Accepted layouts for the current 30-feature model:

```text
feature_1, ..., feature_30
id, feature_1, ..., feature_30
```

Both a normal text header and no header are accepted. The leading ID is
optional and is ignored when 31 columns are present.

Do not remove the label from generated `data/test.csv` and pass it directly
to prediction: those test features are already scaled, so doing so would scale
them a second time. Use raw feature values from the original data schema, as
in `data/example.csv`.

The prediction output has only one column:

```csv
predict
B
M
```

It does not retain the input ID, include probabilities, compare labels, or
report accuracy. Accuracy must be calculated separately when hidden labels are
available.

The checked working command is:

```bash
python mlp.py --predict \
  --input_csv data/example.csv \
  --output_csv data/predictions.csv \
  --model output/saved_model.json \
  --scaler output/scaler.json
```

## Prediction failure and edge-case matrix

| Case | Current result |
| --- | --- |
| Missing input CSV | `Error: [Errno 2] No such file or directory: '...'` |
| Read-protected input, model, or scaler | `Error: [Errno 13] Permission denied: '...'` |
| Invalid JSON model | Raw JSON parsing error is printed as `Error: ...` |
| Model missing `input_features`, `hidden_layers`, or `layers` | Raw `KeyError`, for example `Error: 'input_features'` |
| Model has missing/extra layers or incompatible weight shapes | No early validation. Missing layers can remain randomly initialized, extra layers can be ignored, and bad shapes fail later during matrix multiplication. |
| Scaler missing `min` or `max` | Raw `KeyError`, for example `Error: 'min'` |
| Scaler has the wrong number of values | NumPy broadcasting error during scaling |
| 29 numeric columns | Clear wrong-column-count error |
| 31 numeric columns | Treated as ID plus 30 features |
| ID plus only 29 features | Unsafe: it is indistinguishable from 30 feature columns, so the ID is used as feature 1 without an error. |
| Generated `data/test.csv` / `data/train.csv` | Rejected because their second column is a `B`/`M` label, not a numeric feature. |
| A nonnumeric value after the first row | `Error: Prediction CSV must contain only numeric IDs and features` |
| A bad first data row | Unsafe: it is interpreted as a header and silently discarded. |
| Empty or header-only prediction file | Empty input can fail during reading; a header-only file can write an output containing only the `predict` header. |
| Missing prediction-output parent directory | `Error: Cannot save file into a non-existent directory: '...'` |
| Non-writable prediction-output directory | `Error: [Errno 13] Permission denied: '...'` |

Permission errors are **not ignored**. They bubble up to `main()`, which prints
`Error: ...` and exits with status 1. There is no preflight permission check or
friendlier recovery message.

Source: [src/predict.py](src/predict.py) and [mlp.py](mlp.py).

## CLI failures and documentation differences

Exactly one execution mode is required:

```bash
python mlp.py
# error: choose one execution mode: --split, --train, or --predict

python mlp.py --split --train
# error: choose only one execution mode at a time
```

Options are mode-specific. These documented-looking commands currently fail:

```bash
python mlp.py --split --dataset data/data.csv --seed 42
# error: --seed is only valid with --train

python mlp.py --predict --dataset data/test.csv
# error: --dataset is only valid with --split
```

Use `--input_csv` for prediction. The project does not currently support
evaluating a labelled test file with `--predict`, despite the CLI help and
README wording suggesting that it does.

Additional parser edge case: a negative value supplied as a separate argument
(for example, `--epochs -1`) is mistaken for an option. The `--epochs=-1`
form passes parsing but is still an invalid training configuration.

## Training and split failure cases

### Required training-file layout

The loader expects a headerless CSV in this order:

```text
id, diagnosis, feature_1, ..., feature_N
```

Generated train/test files use this layout. A label-last file is read in the
wrong order without a clear error. The loader does not enforce exactly 30
features before training.

### Data validation gaps

| Case | Current result |
| --- | --- |
| Missing or malformed train/test CSV | OS, pandas, or indexing error is printed as `Error: ...` |
| Non-numeric or missing feature row | Row is silently removed. If every row is removed: `Error: No valid rows remain after cleaning`. |
| Unknown, lowercase, blank, or misplaced label | Not rejected. Anything other than exact uppercase `B`/`M` becomes the invalid target `[0, 0]`. |
| Infinite numeric value | May pass cleaning and later produce invalid scaling/results. |
| Train/test feature widths differ | No early check; a matrix multiplication error occurs during the model forward pass. |
| `--ratio 0` | Split fails when trying to calculate minimum/maximum on empty training data. |
| `--ratio 1` or greater | Can produce an empty test set without a useful validation error. |
| `--ratio=-0.1` | Parses with `=` syntax and can make an unintended split. |
| `--ratio nan` | Fails when conversion to an integer is attempted. |

Split has neither seed support nor stratification. Two normal split runs can
produce different train/test files. The training `--seed` affects shuffle order
but does not make the split reproducible.

### Hyperparameter validation gaps

| Case | Current result |
| --- | --- |
| Omitted learning rate | Valid: the default is `0.0314`. |
| `--batch_size 0` | `Error: range() arg 3 must not be zero`. |
| Negative batch size | No batches are processed, leaving an untrained/random model. |
| `--epochs 0` or negative | Training loop is skipped and an untrained model can be saved. |
| Zero, negative, or non-finite learning rate | No proactive validation; learning may not happen or may diverge. |
| Zero or negative hidden-layer size | No proactive validation; later NumPy errors or unusable networks are possible. |

`--scaler` is accepted for training but is currently unused by
`train_model()`. Normal training relies on the already scaled files produced
by `--split`.

## Output directories and overwrite behavior

| Output | Parent directory created automatically? | Notes |
| --- | --- | --- |
| Split train/test CSV | Yes | The parent is created by `Dataset.save_csv()`. |
| Scaler JSON | Yes | The parent is created by `save_scaler()`. |
| Model JSON | Yes | The parent is created by `MultilayerPerceptron.save()`. |
| Learning-curve plot | No | A missing `--plot_out` parent fails before model saving. |
| Prediction CSV | No | Its parent must already exist and be writable. |

Existing output files are overwritten without a confirmation prompt.

## README and project gaps found during the audit

- README split/predict commands described above do not match the CLI.
- README says train/test contain features followed by diagnosis, but actual
  generated files are `id, diagnosis, 30 scaled features`.
- README says the scaler is saved inside the model; normal training leaves
  `model.scaler` as `null`, and prediction always loads the separate scaler
  file.
- README references `create_network`, `predict_from_file`, `docs/README.md`,
  and image files that are not present in this project.
- README describes `predict_proba()` as returning two probabilities per row,
  but current code returns only the malignant probability column.
- There is no automated test suite, pinned dependency lock file, or documented
  Python-version requirement.
- Model/scaler/plot files under `output/` are ignored by Git, so a fresh clone
  may not contain the default prediction artifacts.

## Before-running checklist

1. Use one mode only: `--split`, `--train`, or `--predict`.
2. Ensure every input file exists and is readable.
3. For prediction, use raw numeric data with exactly 30 features, optionally
   preceded by a numeric ID, and use the matching scaler.
4. Ensure the prediction/plot output parent exists and is writable.
5. For split/train labels, use only uppercase `B` or `M`.
6. Use a finite split ratio strictly between 0 and 1.
7. Use positive layer widths, epochs, and batch size, plus a finite positive
   learning rate.
8. Do not hand-edit the model JSON; verify predictions after loading because
   of the current save/load architecture limitation.
