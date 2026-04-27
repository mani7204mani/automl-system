import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

class FeatureEngineering:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.target_column = self.config["features"]["target_column"]

    def load_data(self):
        train_df = pd.read_csv(self.config["data"]["train_data_path"])
        test_df = pd.read_csv(self.config["data"]["test_data_path"])

        return train_df, test_df

    def split_features_target(self, df):
        X = df.drop(columns=[
            self.target_column,
            "RowNumber",
            "CustomerId",
            "Surname"
        ])
        y = df[self.target_column]
        return X, y

    def get_preprocessor(self, X):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

        return preprocessor

    def run(self):
        train_df, test_df = self.load_data()

        X_train, y_train = self.split_features_target(train_df)
        X_test, y_test = self.split_features_target(test_df)

        preprocessor = self.get_preprocessor(X_train)

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # 🔥 SAVE PREPROCESSOR (THIS IS YOUR STEP 1)
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

        print("Feature Engineering Completed")
        print("Preprocessor saved at artifacts/preprocessor.pkl")

        return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


        