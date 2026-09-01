# Multilayer Perceptron — Breast Cancer Classification

A from-scratch MLP project. See `docs/GUIDE.md` for full theory and usage.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.split --dataset data/data.csv --seed 42
python -m src.train --dataset data/data.csv --layer 24 24 --epochs 84
python -m src.predict --dataset data/test.csv --model saved_model.npy
```

## Structure

```
src/          # implementation stubs
docs/         # GUIDE.md + CHECKLIST.md
data/         # dataset
```

No ML libraries allowed. Use numpy/pandas/matplotlib only.
