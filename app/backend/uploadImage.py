from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException
from pydantic import BaseModel

MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "test"
COLLECTION_NAME = "user faces"


class ImageUploadRequest(BaseModel):
    imageUrl: str
    userName: str


async def upload_image(image_url: str, username: str):
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        result = await collection.insert_one({
            "imageUrl": image_url,
            "username": username
        })

        if result.inserted_id:
            return True

    except Exception as e:
        print(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
