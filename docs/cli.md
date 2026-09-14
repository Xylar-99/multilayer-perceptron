# `mlp.py` — command-line entry point

## Goal

This file is the program's front door. It reads the command typed in the
terminal, checks that the options belong together, and starts one workflow.
It does not contain machine-learning mathematics.

## Functions

| Function | Takes | Returns / does | Why it helps |
| --- | --- | --- | --- |
| `build_parser()` | No arguments | An `argparse.ArgumentParser` | Defines every command-line flag and its default value in one place. |
| `parse_arguments(raw_argv=None)` | Optional list such as `['--train', '--epochs', '84']` | Parsed command-line values | Rejects a flag used with the wrong mode, such as `--split --epochs 84`. |
| `run(raw_argv=None)` | Optional command-line argument list | The result of the selected workflow | Calls `split_dataset`, `train_model`, or `predict_from_file`. |
| `main()` | No arguments | Exit status for the terminal | Shows readable unexpected-error messages without a traceback. |

## Modes

| Mode | Main function called | Main result |
| --- | --- | --- |
| `--split` | `split_dataset(...)` | Train CSV, test CSV, and scaler JSON. |
| `--train` | `train_model(...)` | Trained model JSON and learning-curve image. |
| `--predict` | `predict_from_file(...)` | Evaluation metrics or predictions printed to the terminal. |

Example:

```bash
python mlp.py --train --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
```
