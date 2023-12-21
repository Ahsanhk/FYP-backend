from fastapi import Depends, HTTPException
from pymongo import MongoClient
# from passlib.hash import bcrypt


def validate_user(username, password):
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["test"]
        users_collection = db["user"]

        user_data = users_collection.find_one({"username": username})
        print(user_data)

        if user_data is None:
            return False

        stored_password = user_data['password']

        print("checkpost")

        if password == stored_password:
            print("user verifired successfully!!!")
            return True
        else:
            print("password not verifired")
            return False

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
