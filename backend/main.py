import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, create_model
from typing import List, Dict, Type
import tensorflow as tf
import joblib
from pymongo import MongoClient
import datetime
from fastapi.middleware.cors import CORSMiddleware
# =====================================================
# 🚀 FASTAPI APP
# =====================================================
app = FastAPI(
    title="Dengue Early Warning System API",
    description="CNN-LSTM dengue risk forecasting with MongoDB integration",
    version="5.1"
)

origins = [
    "http://127.0.0.1:5500",  # your Live Server address
    "http://localhost:5500",  # optional, in case Live Server uses localhost
    "http://localhost:8000",  # optional, for API testing
    "http://127.0.0.1:3000",  # Live Preview
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 📦 MODEL & SCALER PATH
# =====================================================
MODEL_PATH = "Model/dengue_classification_model.keras"
SCALER_PATH = "Model/scaler_classification.pkl"

# =====================================================
# ⏱ WINDOW SIZE
# =====================================================
WINDOW = 4

# =====================================================
# 🧠 FEATURES
# =====================================================
final_feature_columns = [
    'YEAR', 'MONTH', 'DAY', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'RH', 'SUNSHINE',
    'POPULATION', 'LAND AREA', 'POP_DENSITY',
    'CASES_lag1', 'CASES_lag2', 'CASES_lag3', 'CASES_lag4',
    'RAINFALL_lag1', 'RAINFALL_lag2', 'RAINFALL_lag3', 'RAINFALL_lag4',
    'TMAX_lag1', 'TMAX_lag2', 'TMAX_lag3', 'TMAX_lag4',
    'TMIN_lag1', 'TMIN_lag2', 'TMIN_lag3', 'TMIN_lag4',
    'TMEAN_lag1', 'TMEAN_lag2', 'TMEAN_lag3', 'TMEAN_lag4',
    'RH_lag1', 'RH_lag2', 'RH_lag3', 'RH_lag4',
    'SUNSHINE_lag1', 'SUNSHINE_lag2', 'SUNSHINE_lag3', 'SUNSHINE_lag4',
    'CASES_roll2_mean', 'CASES_roll4_mean', 'CASES_roll2_sum', 'CASES_roll4_sum',
    'RAINFALL_roll2_mean', 'RAINFALL_roll4_mean',
    'RAINFALL_roll2_sum', 'RAINFALL_roll4_sum',
    'TMEAN_roll2_mean', 'TMEAN_roll4_mean', 'TMEAN_roll2_sum',
    'TMEAN_roll4_sum', 'RH_roll2_mean', 'RH_roll4_mean',
    'RH_roll2_sum', 'RH_roll4_sum'
]

# =====================================================
# 🚨 RISK LABELS
# =====================================================
risk_labels = ["Low", "Moderate", "High", "VeryHigh"]

# =====================================================
# 🌆 CITY MAPPING (PRIMARY KEY = LAND AREA)
# =====================================================
land_area_to_city = {
    24.98: "MANILA CITY",
    171.71: "QUEZON CITY",
    55.8: "CALOOCAN CITY",
    32.69: "LAS PINAS CITY",
    21.57: "MAKATI CITY",
    15.71: "MALABON CITY",
    9.29: "MANDALUYONG CITY",
    21.52: "MARIKINA CITY",
    39.75: "MUNTINLUPA CITY",
    8.94: "NAVOTAS CITY",
    46.57: "PARANAQUE CITY",
    13.97: "PASAY CITY",
    48.46: "PASIG CITY",
    10.4: "PATEROS",
    5.95: "SAN JUAN CITY",
    45.21: "TAGUIG CITY",
    47.02: "VALENZUELA CITY"
}

# =====================================================
# 📥 LOAD MODEL & SCALER
# =====================================================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# =====================================================
# 🔌 MONGODB SETUP
# =====================================================
client = MongoClient("mongodb://localhost:27017/")  # replace with your URI
db = client["dengue_db"]
collection = db["forecasts"]

def save_forecast_to_db(city: str, risk_level: str):
    doc = {
        "city": city,
        "risk_level": risk_level,
        "forecast_week": "Next Week",
        "created_at": datetime.datetime.utcnow()
    }
    collection.insert_one(doc)

# =====================================================
# 📌 PYDANTIC MODELS
# =====================================================
feature_fields: Dict[str, Type] = {col: float for col in final_feature_columns}
FeatureInput = create_model("FeatureInput", **feature_fields)

class DengueForecastInput(BaseModel):
    features: List[FeatureInput]

    @field_validator("features")
    @classmethod
    def check_window(cls, v):
        if len(v) != WINDOW:
            raise ValueError(f"Expected {WINDOW} timesteps, got {len(v)}")
        return v

class DengueForecastOutput(BaseModel):
    city: str
    forecast_week: str
    risk_level: str

class CityRequest(BaseModel):
    city: str

# =====================================================
# 🔧 PREPROCESSING
# =====================================================
def preprocess_input(data: DengueForecastInput):
    try:
        records = [step.model_dump() for step in data.features]
        df = pd.DataFrame(records, columns=final_feature_columns)

        land_area = round(df["LAND AREA"].iloc[0], 2)
        city = land_area_to_city.get(land_area, "Unknown City")

        scaled = scaler.transform(df)
        X = scaled.reshape(1, WINDOW, len(final_feature_columns))

        return X, city
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

# =====================================================
# 🔮 FORECAST ENDPOINT
# =====================================================
@app.post("/forecast", response_model=DengueForecastOutput)
async def forecast_next_week(input_data: DengueForecastInput):
    try:
        X, city = preprocess_input(input_data)
        preds = model.predict(X)
        risk = risk_labels[int(np.argmax(preds[0]))]

        save_forecast_to_db(city, risk)

        return DengueForecastOutput(
            city=city,
            forecast_week="Next Week",
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# 🗺️ FETCH LATEST FORECAST FOR ALL CITIES
# =====================================================
@app.get("/api/latest-forecast")
async def get_latest_forecast_per_city():
    """
    Returns the latest risk level for all cities.
    """
    try:
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$city",
                "city": {"$first": "$city"},
                "risk_level": {"$first": "$risk_level"},
                "forecast_week": {"$first": "$forecast_week"},
                "created_at": {"$first": "$created_at"}
            }},
            {"$project": {"_id": 0}}
        ]
        return list(collection.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# 🏙️ FETCH LATEST FORECAST FOR A SINGLE CITY
# =====================================================
@app.post("/get-risk-level")
async def get_risk_level(request: CityRequest):
    """
    Fetch the latest dengue risk level for a specific city.
    """
    city_name = request.city.strip().upper()
    doc = collection.find_one(
        {"city": city_name},
        sort=[("created_at", -1)]
    )
    if not doc:
        raise HTTPException(status_code=404, detail=f"No data found for city: {city_name}")
    
    return {
        "city": city_name,
        "risk_level": doc.get("risk_level", "Unknown"),
        "forecast_week": doc.get("forecast_week", "Unknown")
    }

# =====================================================
# ❤️ HEALTH CHECK
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "OK",
        "model_loaded": True,
        "mongodb_connected": True,
        "window_size": WINDOW,
        "num_features": len(final_feature_columns),
        "city_detected_from_land_area": True
    }
