from pymongo import MongoClient
from bson.json_util import dumps
from fastapi import HTTPException

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "test"
COLLECTION_NAME = "user"
collection_cards = "user_cards"


def get_user_by_username(username: str):
    try:
        client = MongoClient(MONGODB_URI)

        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        user_data = collection.find_one({"username": username})

        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_json = dumps(user_data)
        return user_json

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}")
