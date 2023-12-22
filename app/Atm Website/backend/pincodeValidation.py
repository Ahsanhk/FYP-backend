from fastapi import Depends, HTTPException
from pymongo import MongoClient
from passlib.hash import bcrypt
from usernameAuth import get_user_by_username


def validate_pincode(username, pincode):
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["test"]
        users_collection = db["user"]

        user_data = users_collection.find_one({"username": username})

        # if username not in user_data:
        #     raise HTTPException(status_code=404, detail="User not found")

        stored_pincode = user_data['pincode']

        if pincode == stored_pincode:
            print("pincode verifired")
            return True
        else:
            print("pincode not verifired")
            return False
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
