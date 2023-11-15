from fastapi import HTTPException, Body
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pymongo import MongoClient


class UserRegistration(BaseModel):
    cnicNumber: str
    email: str
    fullName: str
    mobileNumber: str
    password: str
    pincode: str
    username: str


client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection = db["user"]


async def register_user(user_data: UserRegistration):
    try:
        print("Storing user data in MongoDB:", user_data.dict())
        result = collection.insert_one(user_data.dict())
        user_id = str(result.inserted_id)
        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        print(f"Exception register user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
