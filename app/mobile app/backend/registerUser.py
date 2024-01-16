from fastapi import HTTPException, Body
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pymongo import MongoClient
import bcrypt


class UserRegistration(BaseModel):
    cnicNumber: str
    email: str
    fullName: str
    mobileNumber: str
    password: str
    username: str


client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection = db["user"]


async def register_user(user_data: UserRegistration):
    try:
        hashed_password_bytes = bcrypt.hashpw(
            user_data.password.encode(), bcrypt.gensalt())
        hashed_password_str = hashed_password_bytes.decode('utf-8')
        user_data_dict = user_data.dict()
        user_data_dict["password"] = hashed_password_str

        result = collection.insert_one(user_data_dict)
        user_id = str(result.inserted_id)

        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        print(f"Exception register user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
