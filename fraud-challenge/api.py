"""FastAPI service exposing the fraud model (Module 6).

    uvicorn api:app --reload
    -> interactive docs at http://127.0.0.1:8000/docs

Design note: velocity features (transactions in the last 1h/24h, amount vs 30-day mean, seconds since the
previous transaction) need the card's history. In production they come from a feature store / cache, so the
caller sends them with the transaction. Time, distance and age features are computed here.
"""
import json
from datetime import date, datetime
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config as C
from src import explain as E
from src import features as F
from src import models as MD

app = FastAPI(title="Explainable Fraud Detection API", version="1.0")

_state = {}


def _load():
    if _state:
        return _state
    try:
        bundle = joblib.load(C.ART_DIR / "xgb.joblib")
        thr = json.loads((C.ART_DIR / "thresholds.json").read_text())
        _state.update(
            model=bundle["model"], beta=bundle["beta"], strategy=bundle["strategy"],
            names=json.loads((C.ART_DIR / "feature_names.json").read_text()),
            categories=json.loads((C.ART_DIR / "categories.json").read_text()),
            tau=thr["models"]["xgb"]["tau"], thr=thr,
        )
        _state["explainer"] = E.get_explainer(_state["model"])
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model artifacts missing ({e.filename}). Run steps 1-4 first.")
    return _state


class Transaction(BaseModel):
    ts: datetime = Field(..., examples=["2020-09-14T23:42:10"])
    amt: float = Field(..., gt=0, examples=[742.5])
    category: str = Field(..., examples=["shopping_net"])
    gender: str = Field(..., examples=["F"])
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int = Field(..., ge=0)
    dob: date
    # velocity features supplied by the upstream feature store (neutral defaults = no history)
    n_tx_1h: int = 0
    n_tx_24h: int = 0
    amt_ratio_30d: float = 1.0
    secs_since_prev: float = float(F.CAP_SECS)


class ReasonCode(BaseModel):
    feature: str
    label: str
    value: float
    shap: float
    text: str


class Prediction(BaseModel):
    fraud_score: float
    alert: bool
    threshold: float
    reason_codes: List[ReasonCode]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_state) or (C.ART_DIR / "xgb.joblib").exists()}


@app.get("/model-info")
def model_info():
    s = _load()
    return {"model": "XGBoost", "imbalance_strategy": s["strategy"], "n_features": len(s["names"]),
            "features": s["names"], "threshold": s["tau"], "alert_cost": s["thr"]["alert_cost"],
            "alerts_per_day_budget": s["thr"]["alerts_per_day"]}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction, top_k: Optional[int] = 5):
    s = _load()
    df = pd.DataFrame([tx.model_dump()])
    df["ts"], df["dob"] = pd.to_datetime(df["ts"]), pd.to_datetime(df["dob"])
    X = pd.concat([F.static_features(df, s["categories"]), df[F.BEHAV]], axis=1)[s["names"]].astype(float)
    p = float(s["model"].predict_proba(X)[0, 1])
    if s["beta"] < 1:
        p = float(MD.correct_proba(p, s["beta"]))
    sv = s["explainer"].shap_values(X)[0]
    codes = E.reason_codes(sv, X.iloc[0].to_numpy(), s["names"], top=max(3, min(top_k or 5, 5)))
    return Prediction(fraud_score=p, alert=p >= s["tau"], threshold=s["tau"], reason_codes=codes)
