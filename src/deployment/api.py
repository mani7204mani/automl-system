from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import logging

app = FastAPI()

# -------------------------
# 🔥 Logging (NEW)
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# 🔥 Load artifacts
# -------------------------
model = joblib.load("artifacts/best_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# -------------------------
# 🔥 Columns
# -------------------------
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

MODEL_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# -------------------------
# 🔥 Pydantic Model (FIXED)
# -------------------------
class InputData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# -------------------------
# 🔥 Health endpoint (NEW)
# -------------------------
@app.get("/health")
def health():
    return {"status": "running"}


# -------------------------
# 🔥 Home
# -------------------------
@app.get("/")
def home():
    return {"message": "AutoML API is running"}


# -------------------------
# 🔥 Predict (FIXED)
# -------------------------
@app.post("/predict")
def predict(data: InputData):   # ✅ USING PYDANTIC NOW
    try:
        logger.info(f"Incoming data: {data}")

        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

        # Ensure correct order
        df = df[MODEL_COLUMNS]

        # Transform
        transformed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(transformed)

        logger.info(f"Prediction: {prediction[0]}")

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}