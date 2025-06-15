from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import ipaddress
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Allow all origins; replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model, preprocessing pipeline, and label encoder
model = joblib.load("network.pkl")
preprocess = joblib.load("preprocess.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Network Issue Detection API")

# Pydantic model for request validation
class NetworkInput(BaseModel):
    timestamp: str
    source_ip: str
    dest_ip: str
    protocol: str
    packet_size: int
    latency_ms: float
    error_rate: float
    device_type: str

# Inference function (from your code)
def predict_network_issue(input_dict, model, preprocess, label_encoder):
    df = pd.DataFrame([input_dict])
    df["unix_time"] = pd.to_datetime(df["timestamp"]).astype("int64") // 1_000_000_000
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["minute"] = pd.to_datetime(df["timestamp"]).dt.minute
    df["source_ip_int"] = df["source_ip"].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
    df["dest_ip_int"] = df["dest_ip"].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
    df = df.drop(columns=["timestamp", "source_ip", "dest_ip"])
    X_transformed = preprocess.transform(df)
    pred = model.predict(X_transformed)
    return label_encoder.inverse_transform(pred)[0]

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is live and healthy."}

# Prediction endpoint
@app.post("/predict")
def predict(sample: NetworkInput):
    prediction = predict_network_issue(sample.dict(), model, preprocess, label_encoder)
    return {"predicted_issue_type": prediction}