# Code guide

These guides explain the learning code one file at a time. They focus on what
each class or function is for, what it receives, and what it returns or saves.

```text
raw CSV
  -> data.py: read and clean rows
  -> split.py: create scaled train/test CSVs and a scaler
  -> train.py + model.py: build and train the ReLU + softmax network
  -> predict.py: load the saved model and classify new rows
```

The classifier has two output classes:

| Output index | Diagnosis | Softmax probability |
| --- | --- | --- |
| `0` | benign (`B`) | `P(benign)` |
| `1` | malignant (`M`) | `P(malignant)` |

Softmax makes the two output probabilities add up to `1`. The predicted class
is the index with the larger probability.

## Guides

- [CLI: `mlp.py`](cli.md)
- [Data: `src/data.py`](data.md)
- [Split: `src/split.py`](split.md)
- [Model: `src/model.py`](model.md)
- [Training: `src/train.py`](train.md)
- [Prediction: `src/predict.py`](predict.md)
