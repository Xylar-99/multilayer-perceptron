from .data import Dataset


class DataSplitter:

    @staticmethod
    def preprocess(dataset_path):

        dataset = Dataset.load_csv(dataset_path)
        dataset.cleanup()
        return dataset

    @staticmethod
    def run(dataset_path, train_out, test_out, scaler_out, ratio):

        # 1. Load + cleanup
        dataset = DataSplitter.preprocess(dataset_path)

        # 2. Split
        train_data, test_data = dataset.split(train_ratio=ratio)

        # 3. Fit scaler ONLY on train
        scaler = train_data.fit_scaler()

        # Save scaler to output/scaler.json for prediction and testing
        Dataset.save_scaler(scaler, scaler_out)

        # 4. Scale train and test using SAME scaler
        train_data.scale(scaler)
        test_data.scale(scaler)

        # 5. Save
        train_data.save_csv(train_out)
        test_data.save_csv(test_out)