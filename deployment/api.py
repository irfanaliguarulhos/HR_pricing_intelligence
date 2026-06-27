from pathlib import Path
import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.damohr.features import build_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pricing_classifier.pkl"
app = FastAPI(title="DamoHR Pricing Intelligence API")


class PricingRequest(BaseModel):
    service_line: Literal["Recruiting", "Payroll", "HR Analytics", "Benefits Admin", "Workforce Planning"]
    consultant_seniority: Literal["Junior", "Mid", "Senior", "Principal"]
    region: Literal["Northeast", "Midwest", "South", "West", "Remote"]
    client_size: Literal["SMB", "MidMarket", "Enterprise"]
    contract_months: int
    skill_score: float
    urgency_score: int
    utilization_rate: float
    market_rate: float
    cost_rate: float
    proposed_rate: float
    discount_pct: float = 0.0
    win_probability: float = 0.5


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: PricingRequest):
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([payload.model_dump()])
    df["deal_id"] = "api-request"
    df["proposed_margin"] = (df["proposed_rate"] - df["cost_rate"]) / df["proposed_rate"]
    df["pricing_decision"] = "review"
    features = build_features(df).drop(columns=["pricing_decision", "deal_id"])
    prediction = model.predict(features)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0].round(4)))
    return {"prediction": prediction, "probabilities": probabilities}
