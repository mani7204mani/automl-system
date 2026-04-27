import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score
import joblib
import os
class ModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.problem_type = self.config["features"]["problem_type"]
        
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

    def get_models(self):
        if self.problem_type == "classification":
            return {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest_tuned": "tuned_rf",
                "xgboost_tuned": "tuned_xgb"
            }
        else:
            raise Exception("Only classification supported for now")

    def train_and_evaluate(self, models, X_train, y_train, X_test, y_test):
        best_model = None
        best_score = 0
        best_model_name = ""
        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                if model == "tuned_rf":
                    model = self.tune_random_forest(X_train, y_train)
                    mlflow.log_param("tuning", "optuna_rf")
                elif model == "tuned_xgb":
                    model = self.tune_xgboost(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                preds = model.predict(X_test)
                print("Predictions sample:", preds[:20])
                print("Actual sample:", y_test[:20].values)
                accuracy = accuracy_score(y_test, preds)
                precision = precision_score(y_test, preds,average="weighted")
                recall = recall_score(y_test, preds,average="weighted")
                f1 = f1_score(y_test, preds)

                # Logging
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"{name}: {f1}")

                # Track best model
                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_model_name = name

        print(f"\nBest Model: {best_model_name} with accuracy {best_score}")
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(best_model, "artifacts/best_model.pkl")

        print("Best model saved at artifacts/best_model.pkl")

        return best_model

    def run(self, X_train, y_train, X_test, y_test):
        models = self.get_models()
        best_model = self.train_and_evaluate(models, X_train, y_train, X_test, y_test)
        return best_model
    def tune_random_forest(self, X_train, y_train):
        def objective(trial):

            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 20)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced"
                )
            score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("Best RF Params:", study.best_params)

        best_model = RandomForestClassifier(
                **study.best_params,
                class_weight="balanced"
            )
        best_model.fit(X_train, y_train)

        return best_model
    from xgboost import XGBClassifier

    def tune_xgboost(self, X_train, y_train):
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)
            }

            model = XGBClassifier(
    **params,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("Best XGB Params:", study.best_params)

        best_model = XGBClassifier(
    **study.best_params,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)
        best_model.fit(X_train, y_train)

        return best_model

    