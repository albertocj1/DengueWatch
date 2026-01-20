from pymongo import MongoClient
import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["dengue_db"]
alerts_collection = db["alert_recommendations"]

doc = {
    "city": "TEST CITY",
    "risk_level": "Low",
    "risk_assessment": "Testing",
    "actions": ["Do this", "Do that"],
    "created_at": datetime.datetime.utcnow()
}

result = alerts_collection.insert_one(doc)
print("Inserted ID:", result.inserted_id)

for doc in alerts_collection.find({}, {"_id": 0}):
    print(doc)

