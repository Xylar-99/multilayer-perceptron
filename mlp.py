import sys
import argparse

from src.split import DataSplitter
from src.train import ModelTrainer
from src.predict import ModelPredictor


class MLPApp:
    """Main application dispatcher for all three execution modes with strict argument validation."""

    # Define allowed flags for each mode
    ALLOWED_ARGS = {
        "split": {
            "--split", "--dataset", "--train_out", "--test_out", "--ratio", "--seed", "--help"
        },
        "train": {
            "--train", "--dataset", "--val_dataset", "--layer", "--epochs",
            "--batch_size", "--learning_rate", "--model", "--model_out",
            "--plot_out", "--seed", "--help"
        },
        "predict": {
            "--predict", "--dataset", "--model", "--model_out", "--help"
        }
    }

    # Which mode an argument primarily belongs to (for clear error messages)
    ARG_MODES = {
        "--train_out": "split",
        "--test_out": "split",
        "--ratio": "split",
        "--val_dataset": "train",
        "--layer": "train",
        "--epochs": "train",
        "--batch_size": "train",
        "--learning_rate": "train",
        "--plot_out": "train",
        "--model": "predict",
    }

    @classmethod
    def validate_args(cls, raw_argv):
        """
        Validates that arguments match the chosen execution mode.
        Throws a clear error if an argument from another mode is used.
        """
        # Find all flags starting with '-'
        passed_flags = [arg.split('=')[0] for arg in raw_argv if arg.startswith('--')]

        # Allow help flag
        if "--help" in passed_flags:
            return

        # Check mode selection
        modes_selected = []
        if "--split" in passed_flags:
            modes_selected.append("split")
        if "--train" in passed_flags:
            modes_selected.append("train")
        if "--predict" in passed_flags:
            modes_selected.append("predict")

        if len(modes_selected) == 0:
            print("Error: No execution mode specified. Please choose one: --split, --train, or --predict.\n", file=sys.stderr)
            cls.build_parser().print_help()
            sys.exit(1)

        if len(modes_selected) > 1:
            modes_str = " and ".join([f"'--{m}'" for m in modes_selected])
            print(f"Error: Conflicting modes selected ({modes_str}). Please select only ONE mode at a time.", file=sys.stderr)
            sys.exit(1)

        active_mode = modes_selected[0]
        allowed = cls.ALLOWED_ARGS[active_mode]

        # Check if any passed flag is not allowed in active mode
        for flag in passed_flags:
            if flag not in allowed:
                target_mode = cls.ARG_MODES.get(flag)
                if target_mode:
                    print(f"Error: Argument '{flag}' belongs to '{target_mode}' mode, not allowed with '--{active_mode}'.", file=sys.stderr)
                else:
                    print(f"Error: Argument '{flag}' is not valid with '--{active_mode}'.", file=sys.stderr)
                sys.exit(1)

    @staticmethod
    def build_parser():
        parser = argparse.ArgumentParser(
            description="Multilayer Perceptron (MLP) — Wisconsin Breast Cancer Classification",
            formatter_class=argparse.RawTextHelpFormatter
        )

        # Execution mode flags
        action_group = parser.add_argument_group("Execution Modes (Choose exactly one)")
        action_group.add_argument("--split", action="store_true", help="Part 1: Split dataset into train and test sets")
        action_group.add_argument("--train", action="store_true", help="Part 2: Train neural network")
        action_group.add_argument("--predict", action="store_true", help="Part 3: Evaluate predictions on a dataset")

        # Split Options
        split_group = parser.add_argument_group("Options for --split")
        split_group.add_argument("--dataset", type=str, default="data/data.csv", help="Path to raw CSV dataset")
        split_group.add_argument("--train_out", type=str, default="data/train.csv", help="Destination path for train CSV")
        split_group.add_argument("--test_out", type=str, default="data/test.csv", help="Destination path for test CSV")
        split_group.add_argument("--scaler_out", type=str, default="output/scaler.json", help="Destination path for scaler parameters")
        split_group.add_argument("--ratio", type=float, default=0.8, help="Training split ratio (default: 0.8)")
        split_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling")

        # Train Options
        train_group = parser.add_argument_group("Options for --train")
        train_group.add_argument("--train_data", type=str, default="data/train.csv", help="Path to training CSV dataset")
        train_group.add_argument("--test_data", type=str, default="data/test.csv", help="Path to validation CSV dataset")
        train_group.add_argument("--layer", type=int, nargs="+", default=[24, 24], help="Hidden layer units (default: 24 24)")
        train_group.add_argument("--epochs", type=int, default=84, help="Number of training epochs (default: 84)")
        train_group.add_argument("--batch_size", type=int, default=8, help="Mini-batch size (default: 8)")
        train_group.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate (default: 0.0314)")
        train_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training (default: 42)")
        train_group.add_argument("--model_out", type=str, default="output/saved_model.json", help="Path to save trained model")
        train_group.add_argument("--plot_out", type=str, default="output/learning_curves.png", help="Path to save learning curves plot")

        # Predict Options
        predict_group = parser.add_argument_group("Options for --predict")
        predict_group.add_argument("--model", type=str, default="output/saved_model.json", help="Path to saved model file")
        predict_group.add_argument("--scaler", type=str, default="output/scaler.json", help="Path to scaler parameters")

        return parser

    @classmethod
    def run(cls, raw_argv=None):
        if raw_argv is None:
            raw_argv = sys.argv[1:]

        # Validate arguments strictly against selected mode
        cls.validate_args(raw_argv)

        parser = cls.build_parser()
        parsed_args = parser.parse_args(raw_argv)

        # Dispatch
        if parsed_args.split:
            DataSplitter.run(
                dataset_path=parsed_args.dataset,
                train_out=parsed_args.train_out,
                test_out=parsed_args.test_out,
                scaler_out=parsed_args.scaler_out,
                ratio=parsed_args.ratio,
                seed=parsed_args.seed
            )
        elif parsed_args.train:
            ModelTrainer.run(
                train_data=parsed_args.train_data,
                test_data=parsed_args.test_data,
                hidden_layers=parsed_args.layer,
                epochs=parsed_args.epochs,
                batch_size=parsed_args.batch_size,
                learning_rate=parsed_args.learning_rate,
                model_out=parsed_args.model_out,
                plot_out=parsed_args.plot_out,
                seed=parsed_args.seed,
            )
        elif parsed_args.predict:
            ModelPredictor.run(
                dataset_path=parsed_args.dataset,
                model_path=parsed_args.model
            )


def main():
    try:
        MLPApp.run()
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
