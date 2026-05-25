from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("model/model.pkl")

# Initialize app
app = FastAPI(title="Invoice Expense Classifier API")

# Request schema
class InvoiceRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InvoiceRequest):
    text = data.text.lower()

    prediction = model.predict([text])[0]

    return {
        "category": prediction
    }