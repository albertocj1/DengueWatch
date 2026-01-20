from pymongo import MongoClient
from datetime import datetime
import random

# -------------------------------
# Connect to MongoDB
# -------------------------------
client = MongoClient("mongodb://localhost:27017/")  # Change URI if needed
db = client["dengue_db"]  # Replace with your DB name
alerts_collection = db["forecasts"]

# -------------------------------
# NCR Cities
# -------------------------------
cities = [
    "CALOOCAN CITY", "LAS PINAS CITY", "MAKATI CITY", "MALABON CITY",
    "MANDALUYONG CITY", "MANILA CITY", "MARIKINA CITY", "MUNTINLUPA CITY",
    "NAVOTAS CITY", "PARANAQUE CITY", "PASAY CITY", "PASIG CITY",
    "PATEROS", "QUEZON CITY", "SAN JUAN CITY", "TAGUIG CITY",
    "VALENZUELA CITY"
]

# -------------------------------
# Possible risk levels and forecast weeks
# -------------------------------
risk_levels = ["Low", "Moderate", "High", "VeryHigh"]
forecast_weeks = ["This Week", "Next Week", "Next 2 Weeks"]

# -------------------------------
# Create randomized documents
# -------------------------------
alerts_data = []
for city in cities:
    alert = {
        "city": city,
        "risk_level": random.choice(risk_levels),
        "forecast_week": random.choice(forecast_weeks),
        "created_at": datetime.utcnow()
    }
    alerts_data.append(alert)

# -------------------------------
# Insert into MongoDB
# -------------------------------
result = alerts_collection.insert_many(alerts_data)
print(f"Inserted {len(result.inserted_ids)} documents into the 'alerts' collection.")
