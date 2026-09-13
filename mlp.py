import argparse
import sys

from src.predict import ModelPredictor
from src.split import DataSplitter
from src.train import ModelTrainer


class MLPApp:
    """Parse one CLI mode and send its options to the matching workflow."""

    ALLOWED_ARGS = {
        "split": {
            "--split",
            "--dataset",
            "--train_out",
            "--test_out",
            "--scaler_out",
            "--ratio",
            "--seed",
            "--help",
            "-h",
        },
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
            "--help",
            "-h",
        },
        "predict": {
            "--predict",
            "--dataset",
            "--model",
            "--scaler",
            "--help",
            "-h",
        },
    }

    # This map lets a validation error point to the mode that owns the flag.
    ARG_MODES = {
        "--train_out": "split",
        "--test_out": "split",
        "--scaler_out": "split",
        "--ratio": "split",
        "--train_data": "train",
        "--test_data": "train",
        "--layer": "train",
        "--epochs": "train",
        "--batch_size": "train",
        "--learning_rate": "train",
        "--model_out": "train",
        "--plot_out": "train",
        "--model": "predict",
    }

    @classmethod
    def run(cls, raw_argv=None):
        """Validate arguments, parse them, and run the requested workflow."""
        if raw_argv is None:
            raw_argv = sys.argv[1:]

        cls.validate_args(raw_argv)
        parsed_args = cls.build_parser().parse_args(raw_argv)
        cls._run_selected_mode(parsed_args)

    @classmethod
    def validate_args(cls, raw_argv):
        """Ensure that exactly one mode was selected and owns every option."""
        passed_flags = cls._get_passed_flags(raw_argv)
        if cls._help_was_requested(passed_flags):
            return

        active_mode = cls._get_selected_mode(passed_flags)
        cls._validate_mode_arguments(active_mode, passed_flags)

    @staticmethod
    def build_parser():
        """Build the command-line parser used by all three workflows."""
        parser = argparse.ArgumentParser(
            description="Multilayer Perceptron (MLP) — Wisconsin Breast Cancer Classification",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        mode_group = parser.add_argument_group("Execution modes (choose exactly one)")
        mode_group.add_argument(
            "--split",
            action="store_true",
            help="Split the raw dataset into training and test files.",
        )
        mode_group.add_argument(
            "--train",
            action="store_true",
            help="Train a neural network.",
        )
        mode_group.add_argument(
            "--predict",
            action="store_true",
            help="Evaluate a labeled file or predict feature-only rows.",
        )

        split_group = parser.add_argument_group("Options for --split")
        split_group.add_argument(
            "--dataset",
            type=str,
            default="data/data.csv",
            help="Raw CSV for --split, or data to evaluate for --predict.",
        )
        split_group.add_argument(
            "--train_out",
            type=str,
            default="data/train.csv",
            help="Destination path for the training CSV.",
        )
        split_group.add_argument(
            "--test_out",
            type=str,
            default="data/test.csv",
            help="Destination path for the test CSV.",
        )
        split_group.add_argument(
            "--scaler_out",
            type=str,
            default="output/scaler.json",
            help="Destination path for train-set scaler parameters.",
        )
        split_group.add_argument(
            "--ratio",
            type=float,
            default=0.8,
            help="Training split ratio (default: 0.8).",
        )

        train_group = parser.add_argument_group("Options for --train")
        train_group.add_argument(
            "--train_data",
            type=str,
            default="data/train.csv",
            help="Path to the training CSV.",
        )
        train_group.add_argument(
            "--test_data",
            type=str,
            default="data/test.csv",
            help="Path to the test CSV used after training.",
        )
        train_group.add_argument(
            "--layer",
            type=int,
            nargs="+",
            default=[24, 24],
            help="Units in each hidden layer (default: 24 24).",
        )
        train_group.add_argument(
            "--epochs",
            type=int,
            default=84,
            help="Number of training epochs (default: 84).",
        )
        train_group.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Mini-batch size (default: 8).",
        )
        train_group.add_argument(
            "--learning_rate",
            type=float,
            default=0.0314,
            help="Learning rate (default: 0.0314).",
        )
        train_group.add_argument(
            "--model_out",
            type=str,
            default="output/saved_model.json",
            help="Path for the trained model.",
        )
        train_group.add_argument(
            "--plot_out",
            type=str,
            default="output/learning_curves.png",
            help="Path for the loss and accuracy plot.",
        )

        predict_group = parser.add_argument_group("Options for --predict")
        predict_group.add_argument(
            "--model",
            type=str,
            default="output/saved_model.json",
            help="Path to the saved model.",
        )

        shared_group = parser.add_argument_group("Options shared by modes")
        shared_group.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for --split and --train (default: 42).",
        )
        shared_group.add_argument(
            "--scaler",
            type=str,
            default=None,
            help=(
                "Optional scaler JSON for --train or --predict. "
                "Training stores it in the model; prediction uses it as an override."
            ),
        )

        return parser


    @classmethod
    def _run_selected_mode(cls, parsed_args):
        """Dispatch parsed options to the workflow selected by its mode flag."""
        if parsed_args.split:
            cls._run_split(parsed_args)
        elif parsed_args.train:
            cls._run_train(parsed_args)
        elif parsed_args.predict:
            cls._run_predict(parsed_args)

    @staticmethod
    def _run_split(parsed_args):
        """Run dataset preparation and splitting."""
        DataSplitter.run(
            dataset_path=parsed_args.dataset,
            train_out=parsed_args.train_out,
            test_out=parsed_args.test_out,
            scaler_out=parsed_args.scaler_out,
            ratio=parsed_args.ratio,
            seed=parsed_args.seed,
        )

    @staticmethod
    def _run_train(parsed_args):
        """Run the model training workflow."""
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
            scaler_path=parsed_args.scaler,
        )

    @staticmethod
    def _run_predict(parsed_args):
        """Run model evaluation or feature-only prediction."""
        ModelPredictor.run(
            dataset_path=parsed_args.dataset,
            model_path=parsed_args.model,
            scaler_path=parsed_args.scaler,
        )

    @staticmethod
    def _get_passed_flags(raw_argv):
        """Return only option names, including ``--option=value`` forms."""
        passed_flags = []
        for argument in raw_argv:
            is_long_option = argument.startswith("--")
            is_short_help_option = argument == "-h"
            if is_long_option or is_short_help_option:
                passed_flags.append(argument.split("=", maxsplit=1)[0])
        return passed_flags

    @staticmethod
    def _help_was_requested(passed_flags):
        """Return whether argparse should display its help text immediately."""
        return "--help" in passed_flags or "-h" in passed_flags

    @classmethod
    def _get_selected_mode(cls, passed_flags):
        """Return the only selected mode, or stop with a clear CLI error."""
        modes_selected = []
        for mode in ("split", "train", "predict"):
            if f"--{mode}" in passed_flags:
                modes_selected.append(mode)

        if not modes_selected:
            cls._exit_for_missing_mode()

        if len(modes_selected) > 1:
            cls._exit_for_conflicting_modes(modes_selected)

        return modes_selected[0]

    @classmethod
    def _validate_mode_arguments(cls, active_mode, passed_flags):
        """Reject options that do not belong to the selected workflow."""
        allowed_flags = cls.ALLOWED_ARGS[active_mode]
        for flag in passed_flags:
            if flag not in allowed_flags:
                cls._exit_for_invalid_argument(flag, active_mode)

    @classmethod
    def _exit_for_missing_mode(cls):
        """Print guidance when no workflow mode was selected."""
        print(
            "Error: No execution mode specified. "
            "Please choose one: --split, --train, or --predict.\n",
            file=sys.stderr,
        )
        cls.build_parser().print_help()
        sys.exit(1)

    @staticmethod
    def _exit_for_conflicting_modes(modes_selected):
        """Print guidance when more than one workflow mode was selected."""
        modes_text = " and ".join(f"'--{mode}'" for mode in modes_selected)
        print(
            f"Error: Conflicting modes selected ({modes_text}). "
            "Please select only ONE mode at a time.",
            file=sys.stderr,
        )
        sys.exit(1)

    @classmethod
    def _exit_for_invalid_argument(cls, flag, active_mode):
        """Explain why an option cannot be used with the selected mode."""
        target_mode = cls.ARG_MODES.get(flag)
        if target_mode:
            message = (
                f"Error: Argument '{flag}' belongs to '{target_mode}' mode, "
                f"not allowed with '--{active_mode}'."
            )
        else:
            message = f"Error: Argument '{flag}' is not valid with '--{active_mode}'."

        print(message, file=sys.stderr)
        sys.exit(1)


def main():
    """Run the command-line application and present unexpected errors clearly."""
    try:
        MLPApp.run()
    except Exception as error:
        print(f"Error occurred: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
