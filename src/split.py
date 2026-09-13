"""Prepare the raw dataset and save train/test files."""

import numpy as np

from .data import Dataset, save_scaler


def split_dataset(dataset_path, train_out, test_out, scaler_out, ratio=0.8, seed=None):
    """Clean, split, scale, and save the Wisconsin breast-cancer dataset."""
    dataset = Dataset.from_csv(dataset_path).clean()
    random_generator = np.random.default_rng(seed)
    train_data, test_data = dataset.split(ratio, rng=random_generator)

    scaler = train_data.fit_scaler()
    train_data.scale(scaler)
    test_data.scale(scaler)

    train_data.save_csv(train_out)
    test_data.save_csv(test_out)
    save_scaler(scaler, scaler_out)
    return train_data, test_data, scaler
