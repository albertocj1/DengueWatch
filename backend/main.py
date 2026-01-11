import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, create_model
from typing import List, Dict, Type
import tensorflow as tf
import joblib

# =====================================================
# 🚀 FASTAPI APP
# =====================================================
app = FastAPI(
    title="Dengue Early Warning System API",
    description="CNN-LSTM based dengue risk forecasting (Next Week) with raw input",
    version="4.1"
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
risk_labels = ["Low", "Moderate", "High", "Very High"]

# =====================================================
# 🌆 CITY MAPPING BY LAND AREA
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
# 📌 Pydantic Models
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

# =====================================================
# 🔧 PREPROCESSING (raw → scaled)
# =====================================================
def preprocess_input(data: DengueForecastInput):
    try:
        # Convert list of FeatureInput to DataFrame
        records = [step.model_dump() for step in data.features]
        df = pd.DataFrame(records, columns=final_feature_columns)

        # Get city from LAND AREA
        land_area = round(df["LAND AREA"].iloc[0], 2)
        city = land_area_to_city.get(land_area, "Unknown City")

        # Scale using the saved scaler
        scaled = scaler.transform(df)

        # Reshape for CNN-LSTM: (batch_size=1, time_steps=WINDOW, features)
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
        predictions = model.predict(X)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_risk = risk_labels[predicted_index]

        return DengueForecastOutput(
            city=city,
            forecast_week="Next Week",
            risk_level=predicted_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# ❤️ HEALTH CHECK
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "OK",
        "model_loaded": True,
        "window_size": WINDOW,
        "num_features": len(final_feature_columns),
        "input_scaled": False,  # raw input is accepted
        "city_detected_from_land_area": True
    }
