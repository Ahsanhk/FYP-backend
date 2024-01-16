from fastapi import Depends, HTTPException
from pymongo import MongoClient
# from passlib.hash import bcrypt
import bcrypt


def validate_user(username, password):
    try:
        print(username, password)
        client = MongoClient("mongodb://localhost:27017")
        db = client["test"]
        users_collection = db["user"]

        user_data = users_collection.find_one({"username": username})
        print(user_data)

        if user_data is None:
            return False

        stored_password_hash = user_data['password'].encode('utf-8')

        print(stored_password_hash)

        if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
            print("User verified successfully!!!")
            return True
        else:
            print("password not verifired")
            return False

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
