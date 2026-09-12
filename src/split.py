import numpy as np

from .data import Dataset


class DataSplitter:
    """Prepare, split, scale, and save a labeled dataset."""

    @staticmethod
    def run(dataset_path, train_out, test_out, scaler_out, ratio, seed=None):
        """Run the complete dataset preparation workflow."""
        dataset = DataSplitter.preprocess(dataset_path)
        train_data, test_data = DataSplitter._split_dataset(
            dataset,
            ratio,
            seed,
        )

        scaler = train_data.fit_scaler()
        Dataset.save_scaler(scaler, scaler_out)

        DataSplitter._scale_datasets(train_data, test_data, scaler)
        DataSplitter._save_datasets(train_data, test_data, train_out, test_out)

    @staticmethod
    def preprocess(dataset_path):
        """Load a labeled CSV file and remove incomplete rows."""
        dataset = Dataset.load_csv(dataset_path)
        return dataset.cleanup()

    @staticmethod
    def _split_dataset(dataset, ratio, seed):
        """Create a train/test split, optionally with a repeatable shuffle."""
        random_generator = DataSplitter._create_random_generator(seed)
        return dataset.split(train_ratio=ratio, rng=random_generator)

    @staticmethod
    def _scale_datasets(train_data, test_data, scaler):
        """Scale both datasets with parameters fitted only on the train data."""
        train_data.scale(scaler)
        test_data.scale(scaler)

    @staticmethod
    def _save_datasets(train_data, test_data, train_out, test_out):
        """Save the prepared train and test datasets."""
        train_data.save_csv(train_out)
        test_data.save_csv(test_out)

    @staticmethod
    def _create_random_generator(seed):
        """Return a local random generator when a seed was requested."""
        if seed is None:
            return None

        return np.random.default_rng(seed)
