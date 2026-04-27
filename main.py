from src.data.data_ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineering
from src.models.model_trainer import ModelTrainer


def main():
    ingestion = DataIngestion()
    ingestion.run()

    feature_engineering = FeatureEngineering()
    X_train, X_test, y_train, y_test, preprocessor = feature_engineering.run()

    trainer = ModelTrainer()
    best_model = trainer.run(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()