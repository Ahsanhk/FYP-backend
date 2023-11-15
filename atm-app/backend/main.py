import os
from twilio.rest import Client
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
import cv2
from pydantic import BaseModel
import json
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
from firebase_admin import auth, credentials, initialize_app

from uploadImage import upload_image
from handleImage import process_image
from registerUser import register_user, UserRegistration
from userAuth import validate_user
from handleOTP import generate_otp, send_otp_via_sms, otpData

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection = db["user"]

temporary_otp = None

# registers a user


@app.post('/register')
async def handle_registration(user_data: UserRegistration):
    try:
        response_data = await register_user(user_data)

    except HTTPException as e:
        print(f"HTTPException: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        print(f"Exception: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    print("Response Data:", response_data)
    return JSONResponse(content=response_data)

# authentication of the user


@app.get("/authentication")
async def validate_user_route(username: str, password: str):
    print(username, password)
    is_valid = validate_user(username, password)
    print(is_valid)
    if (is_valid):
        return True
    else:
        return False

# to store captured image for recognition on mongodb


@app.post("/upload-image")
async def handle_upload_image(image_url: str, username: str):
    isUploaded = await upload_image(image_url, username)
    print("upload image", isUploaded)


@app.post("/handle_Image")
async def handle_image(username: str, image: UploadFile = File(...)):
    processed_data = process_image(username, image)
    if "error" in processed_data:
        return {"error": processed_data["error"]}
    user_collection.insert_one(processed_data)
    return {"message": "Image processed and stored successfully"}


@app.post("/generate-otp")
def generate_otp_endpoint(mobileNumber: otpData):
    global temporary_otp
    otp = generate_otp()

    try:
        print(mobileNumber)
        send_otp_via_sms(mobileNumber.mobileNumber, otp)
        return otp
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send OTP: {str(e)}")


@app.get("/get-temporary-otp")
def get_temporary_otp():
    global temporary_otp
    print(temporary_otp)
