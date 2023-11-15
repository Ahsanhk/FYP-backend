from fastapi import Depends, HTTPException
from pymongo import MongoClient
# from passlib.hash import bcrypt


def validate_user(username, password):
    print('username in auth: ', username)
    print('password in auth: ', password)
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["test"]
        users_collection = db["user"]

        user_data = users_collection.find_one({"username": username})

        # if username not in user_data:
        #     raise HTTPException(status_code=404, detail="User not found")

        stored_password = user_data['password']

        if password == stored_password:
            print("user verifired successfully!!!")
            return True
        else:
            print("password not verifired")
            return False

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
