# test_all_ncr_cities.py
import requests

# URL of your running FastAPI
URL = "http://127.0.0.1:8000/api/predict"

# NCR cities (must match keys in CITY_DATA of your API)
CITIES = [
    "MANILA CITY", "QUEZON CITY", "CALOOCAN CITY", "LAS PINAS CITY",
    "MAKATI CITY", "MALABON CITY", "MANDALUYONG CITY", "MARIKINA CITY",
    "MUNTINLUPA CITY", "NAVOTAS CITY", "PARANAQUE CITY", "PASAY CITY",
    "PASIG CITY", "SAN JUAN CITY", "TAGUIG CITY", "VALENZUELA CITY",
    "PATEROS"
]

# Example recent_cases for testing (replace with actual last 4 weeks data if available)
RECENT_CASES = [100, 120, 90, 110]

for city in CITIES:
    payload = {
        "city": city,
        "recent_cases": RECENT_CASES
    }

    print(f"\n=== {city} ===")
    try:
        response = requests.post(URL, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            continue

        data = response.json()

        print(f"Latest Risk: {data['latest_risk']}")
        print("Week | Risk      | Confidence")
        print("-" * 30)
        for week in data["weekly_forecast"]:
            print(f"{week['week']:>4} | {week['risk']:<9} | {week['confidence']:.2f}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {city}: {e}")
    except Exception as e:
        print(f"Unexpected error for {city}: {e}")
