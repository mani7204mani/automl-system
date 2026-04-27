import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


class DataIngestion:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def load_data(self):
        path = self.config["data"]["raw_data_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            raise ValueError("Dataset is empty")

        return df

    def split_data(self, df):
        test_size = self.config["data"]["test_size"]
        random_state = self.config["data"]["random_state"]

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        return train_df, test_df

    def save_data(self, train_df, test_df):
        train_path = self.config["data"]["train_data_path"]
        test_path = self.config["data"]["test_data_path"]

        os.makedirs(os.path.dirname(train_path), exist_ok=True)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    def run(self):
        df = self.load_data()
        train_df, test_df = self.split_data(df)
        self.save_data(train_df, test_df)

        print("Data Ingestion Completed")