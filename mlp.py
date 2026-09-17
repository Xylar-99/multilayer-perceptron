"""Command-line entry point for splitting data, training, and prediction."""

import argparse
import sys

from src.predict import Predictor
from src.split import split_dataset
from src.train import train_model


MODE_OPTIONS = {
    "split": {"--split", "--dataset", "--train_out", "--test_out", "--scaler_out", "--ratio"},
    "train": {
        "--train",
        "--train_data",
        "--test_data",
        "--layer",
        "--epochs",
        "--batch_size",
        "--learning_rate",
        "--model_out",
        "--plot_out",
        "--seed",
        "--scaler",
    },
    "predict": {"--predict", "--input_csv", "--output_csv", "--model", "--scaler"},
}


def build_parser():
    """Create the parser while keeping the three workflows visible in its help."""
    parser = argparse.ArgumentParser(
        description="Multilayer Perceptron (MLP) — Wisconsin Breast Cancer Classification"
    )
    mode_group = parser.add_argument_group("Execution modes (choose exactly one)")
    mode_group.add_argument("--split", action="store_true", help="Split and scale the raw dataset.")
    mode_group.add_argument("--train", action="store_true", help="Train a neural network.")
    mode_group.add_argument("--predict", action="store_true", help="Evaluate labeled data or predict feature-only rows.")

    split_group = parser.add_argument_group("Options for --split")
    split_group.add_argument("--dataset", default="data/data.csv", help="Raw CSV for --split or data for --predict.")
    split_group.add_argument("--train_out", default="data/train.csv", help="Destination path for the training CSV.")
    split_group.add_argument("--test_out", default="data/test.csv", help="Destination path for the test CSV.")
    split_group.add_argument("--scaler_out", default="output/scaler.json", help="Destination path for scaler JSON.")
    split_group.add_argument("--ratio", type=float, default=0.8, help="Training split ratio (default: 0.8).")

    train_group = parser.add_argument_group("Options for --train")
    train_group.add_argument("--train_data", default="data/train.csv", help="Path to the training CSV.")
    train_group.add_argument("--test_data", default="data/test.csv", help="Path to the test CSV.")
    train_group.add_argument("--layer", type=int, nargs="+", default=[24, 24], help="Hidden-layer units (default: 24 24).")
    train_group.add_argument("--epochs", type=int, default=84, help="Number of training epochs (default: 84).")
    train_group.add_argument("--batch_size", type=int, default=8, help="Mini-batch size (default: 8).")
    train_group.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate (default: 0.0314).")
    train_group.add_argument("--model_out", default="output/saved_model.json", help="Path for the trained model.")
    train_group.add_argument("--plot_out", default="output/learning_curves.png", help="Path for the learning-curve plot.")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")


    predict_group = parser.add_argument_group("Options for --predict")
    predict_group.add_argument("--model", default="output/saved_model.json", help="Path to the saved model.")
    predict_group.add_argument("--input_csv", default="data/example.csv", help="CSV for --predict (features or labeled).")
    predict_group.add_argument("--output_csv", default="data/predictions.csv", help="Destination path for predictions CSV.")


    shared_group = parser.add_argument_group("Shared options")
    shared_group.add_argument("--scaler", default="output/scaler.json", help="Path to the scaler JSON (used for both --train and --predict).")


    return parser


def parse_arguments(raw_argv=None):
    """Parse one workflow and reject options that belong to a different mode."""
    argv = sys.argv[1:] if raw_argv is None else raw_argv
    parser = build_parser()
    passed_flags = {argument.split("=", maxsplit=1)[0] for argument in argv if argument.startswith("-")}
    if "--help" in passed_flags or "-h" in passed_flags:
        return parser.parse_args(argv)

    selected_modes = [mode for mode in MODE_OPTIONS if f"--{mode}" in passed_flags]
    if not selected_modes:
        parser.error("choose one execution mode: --split, --train, or --predict")
    if len(selected_modes) > 1:
        parser.error("choose only one execution mode at a time")

    active_mode = selected_modes[0]
    for flag in passed_flags:
        if flag not in MODE_OPTIONS[active_mode]:
            owners = [mode for mode, options in MODE_OPTIONS.items() if flag in options]
            if owners:
                parser.error(f"{flag} is only valid with " + " or ".join(f"--{mode}" for mode in owners))
            parser.error(f"unrecognised argument: {flag}")
    return parser.parse_args(argv)


def run(raw_argv=None):
    """Run the CLI workflow selected by the user."""
    args = parse_arguments(raw_argv)
    if args.split:
        return split_dataset(args.dataset, args.train_out, args.test_out, args.scaler_out, args.ratio)
    if args.train:
        return train_model(
            args.train_data,
            args.test_data,
            args.layer,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.model_out,
            args.plot_out,
            args.seed,
            args.scaler,
        )
    return Predictor(args.model, args.scaler , args.input_csv , args.output_csv).predict()


def main():
    """Run the CLI and present unexpected errors without a traceback."""
    try:
        run()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
