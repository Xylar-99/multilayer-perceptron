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
    "predict": {"--predict", "--input_csv", "--dataset", "--output_csv", "--model", "--scaler"},
}


class ModeArgumentParser(argparse.ArgumentParser):
    """Format errors with only the options for the selected mode."""

    def __init__(self, mode, *args, **kwargs):
        self.mode = mode
        self.mode_actions = []
        super().__init__(*args, **kwargs)

    def add_mode_argument(self, group, *args, **kwargs):
        action = group.add_argument(*args, **kwargs)
        self.mode_actions.append(action)
        return action

    def format_mode_options(self):
        formatter = argparse.RawTextHelpFormatter(prog=self.prog)
        formatter.start_section(f"{self.mode.title()} options")
        formatter.add_arguments(self.mode_actions)
        formatter.end_section()
        return formatter.format_help()

    def error(self, message):
        print(f"{self.prog}: error: {message}\n", file=sys.stderr)
        print(f"Valid flags for --{self.mode}:\n", file=sys.stderr)
        print(self.format_mode_options(), end="", file=sys.stderr)
        self.exit(2)


def build_parser():
    """Create the parser used before an execution mode is selected."""
    parser = argparse.ArgumentParser(
        description="Multilayer Perceptron (MLP) — Wisconsin Breast Cancer Classification"
    )
    parser.add_argument("--split", action="store_true", help="Split and scale the raw dataset.")
    parser.add_argument("--train", action="store_true", help="Train a neural network.")
    parser.add_argument("--predict", action="store_true", help="Predict using a trained model.")
    return parser


def build_mode_parser(mode):

    descriptions = {
        "split": "Split and scale the raw dataset.",
        "train": "Train a neural network.",
        "predict": "Predict using a trained model.",
    }
    parser = ModeArgumentParser(
        mode,
        description=descriptions[mode],
        usage=f"%(prog)s --{mode} [OPTIONS]",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    parser.set_defaults(split=False, train=False, predict=False)

    group = parser.add_argument_group(f"{mode.title()} options")

    def add(*args, **kwargs):
        return parser.add_mode_argument(group, *args, **kwargs)

    if mode == "split":
        add("--split", action="store_true", help="Split and scale the dataset.")
        add("--dataset", default="data/data.csv", metavar="DATASET", help="Raw CSV.\nExample: --dataset data/data.csv")
        add("--train_out", default="data/train.csv", help="Destination path for the training CSV.")
        add("--test_out", default="data/test.csv", help="Destination path for the test CSV.")
        add("--scaler_out", default="output/scaler.json", help="Destination path for scaler JSON.")
        add("--ratio", type=float, default=0.8, help="Training split ratio (default: 0.8).")
    elif mode == "train":
        add("--train", action="store_true", help="Train a neural network.")
        add("--train_data", default="data/train.csv", help="Path to the training CSV.")
        add("--test_data", default="data/test.csv", help="Path to the test CSV.")
        add("--layer", type=int, nargs="+", default=[24, 24], help="Hidden-layer units (default: 24 24).")
        add("--epochs", type=int, default=466, help="Number of training epochs (default: 84).")
        add("--batch_size", type=int, default=15, help="Mini-batch size (default: 8).")
        add("--learning_rate", type=float, default=0.0314, help="Learning rate (default: 0.0314).")
        add("--model_out", default="output/saved_model.json", help="Path for the trained model JSON.\nExample: --model_out models/model.json")
        add("--plot_out", default="output/learning_curves.png", help="Path for the learning-curve plot.")
        add("--seed", type=int, default=13, help="Random seed for reproducibility (default: 42).")
        add("--scaler", default="output/scaler.json", help="Path to the scaler JSON.")
    else:
        add("--predict", action="store_true", help="Predict using a trained model.")
        add("--input_csv", "--dataset", dest="input_csv", default="data/example.csv", metavar="INPUT", help="Prediction CSV.\nExample: --dataset data/test.csv")
        add("--output_csv", default="data/predictions.csv", metavar="OUTPUT", help="Destination path for predictions CSV.\nExample: --output_csv data/predictions.csv")
        add("--model", default="output/saved_model.json", metavar="MODEL", help="Saved model JSON.\nExample: --model models/model.json")
        add("--scaler", default="output/scaler.json", metavar="SCALER", help="Saved scaler JSON.\nExample: --scaler models/scaler.json")
        parser.set_defaults(output_csv="data/predictions.csv")

    return parser


def parse_arguments(raw_argv=None):
    """Parse one workflow and reject options that belong to a different mode."""
    argv = sys.argv[1:] if raw_argv is None else raw_argv
    passed_flags = [argument.split("=", maxsplit=1)[0] for argument in argv if argument.startswith("-")]
    parser = build_parser()

    selected_modes = [mode for mode in MODE_OPTIONS if f"--{mode}" in passed_flags]
    if not selected_modes:
        if "--help" in passed_flags or "-h" in passed_flags:
            return parser.parse_args(argv)
        parser.error("choose one execution mode: --split, --train, or --predict")
    if len(selected_modes) > 1:
        parser.error("choose only one execution mode at a time")

    active_mode = selected_modes[0]
    parser = build_mode_parser(active_mode)
    if "--help" in passed_flags or "-h" in passed_flags:
        return parser.parse_args(argv)

    for flag in passed_flags:
        if flag not in MODE_OPTIONS[active_mode]:
            parser.error(f"{flag} is not valid with --{active_mode}")

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
    return Predictor(args.model, args.scaler, args.input_csv, args.output_csv).predict()


def main():
    """Run the CLI and present unexpected errors without a traceback."""
    try:
        run()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
