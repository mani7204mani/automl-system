# 🚀 AutoML Customer Churn Prediction System

## 📌 Overview

This project is an end-to-end Machine Learning system designed to predict customer churn.
It covers the complete ML lifecycle — from data ingestion to deployment — with a production-ready API.

---

## ⚙️ Tech Stack

* **Programming:** Python
* **ML Models:** Scikit-learn, XGBoost
* **Hyperparameter Tuning:** Optuna
* **Experiment Tracking:** MLflow
* **API Deployment:** FastAPI
* **Containerization:** Docker

---

## 🏗️ Architecture

Raw Data → Data Ingestion → Feature Engineering → Model Training → MLflow Tracking → FastAPI → Docker

---

## 📂 Project Structure

```
automl-system/
│
├── artifacts/              # Saved models & preprocessor
├── config/                 # Configuration files
├── data/                   # Dataset (sample/demo)
├── src/
│   ├── data/               # Data ingestion
│   ├── features/           # Feature engineering
│   ├── models/             # Model training
│   ├── deployment/         # FastAPI app
│   ├── pipelines/          # Pipelines
│   └── utils/              # Utility functions
│
├── main.py                 # Entry point
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🔄 Workflow

1. Data ingestion from CSV
2. Train-test split
3. Feature engineering (scaling + encoding)
4. Model training with multiple algorithms
5. Hyperparameter tuning using Optuna
6. Experiment tracking using MLflow
7. Best model selection
8. Deployment using FastAPI
9. Containerization using Docker

---

## ▶️ Run Locally

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run training pipeline

```bash
python main.py
```

### Step 3: Start API

```bash
uvicorn src.deployment.api:app --reload
```

### Step 4: Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

### Build Image

```bash
docker build -t automl-app .
```

### Run Container

```bash
docker run -p 8000:8000 automl-app
```

### Access API

```
http://localhost:8000/docs
```

---

## 📡 API Endpoint

### POST `/predict`

#### Sample Input

```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```

#### Sample Output

```json
{
  "prediction": 1,
  "result": "Churn"
}
```

---

## 📊 Model Details

* Logistic Regression
* Random Forest (Tuned with Optuna)
* XGBoost (Tuned with Optuna)

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📈 Experiment Tracking (MLflow)

MLflow is used to:

* Track experiments
* Compare models
* Log metrics and parameters

Run MLflow UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## ⚠️ Notes

* Dataset included is for demonstration purposes
* Preprocessing pipeline is saved and reused during inference
* Model and preprocessor are stored in `artifacts/`

---

## 🎯 Key Highlights

* End-to-end ML pipeline
* Hyperparameter tuning with Optuna
* Experiment tracking with MLflow
* REST API using FastAPI
* Dockerized deployment

---

## 🚀 Future Improvements

* Add CI/CD pipeline
* Deploy on cloud (AWS/Azure/GCP)
* Add monitoring & logging
* Improve model performance

---

## 👨‍💻 Author

Mani Shankar Reddy P
AI/ML Enthusiast

---
