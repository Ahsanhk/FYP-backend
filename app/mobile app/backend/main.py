# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from bson import json_util
from bson import ObjectId
from typing import Optional
from typing import List
from pydantic import EmailStr, BaseModel
from starlette.responses import JSONResponse
from starlette.requests import Request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
from twilio.rest import Client
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
collection_user = db["user"]
collection_transactions = db["user_transactions"]
collection_images = db["user_images"]
collection_otp = db["user_otp"]
collection_cards = db["user_cards"]


class userData(BaseModel):
    cnicNumber: str
    email: str
    fullName: str
    mobileNumber: str
    password: str
    pincode: str
    username: str


class ProfilePicData(BaseModel):
    imageURL: str
    username: str


@app.post("/signup/")
async def register_user(user_data: userData):
    try:
        print("Storing user data in MongoDB:", user_data.dict())
        result = collection_user.insert_one(user_data.dict())
        user_id = str(result.inserted_id)

        # creating a username instance in transaction dv
        transactions_data = {
            "username": user_data.username,
            "transaction": []
        }
        collection_transactions.insert_one(transactions_data)

        # creating a username instance in images db
        images_data = {
            "username": user_data.username,
            "profilePicture": {},
            "user_faces": [],
        }
        collection_images.insert_one(images_data)

        # creating username instance in otp
        otp_data = {
            "username": user_data.username,
            "otp": {}
        }
        collection_otp.insert_one(otp_data)

        # creating a username instance in cards db
        card_data = {
            "username": user_data.username,
            "cardNumber": {},
            "bankName": {},
            "updatedIssueDate": {},
            "activeStatus": True,
        }
        collection_cards.insert_one(card_data)

        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        print(f"Exception register user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class UploadUserFaceRequest(BaseModel):
    username: str
    imageUrl: str
    faceName: str


@app.post("/upload-user-face")
async def upload_user_face(request_data: UploadUserFaceRequest):
    try:
        username = request_data.username
        imageUrl = request_data.imageUrl
        faceName = request_data.faceName

        user_image_doc = collection_images.find_one({"username": username})

        if user_image_doc:
            new_face_data = {"faceName": faceName, "imageUrl": imageUrl}
            collection_images.update_one(
                {"_id": user_image_doc["_id"]},
                {"$push": {"user_faces": new_face_data}}
            )

            return {"message": f"Image URL added to user_faces for {username}"}
        else:
            return {"message": "Username not found"}
    except Exception as e:
        print(f"Exception uploading user face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-profile-pic")
async def upload_profile_pic(profile_pic_data: ProfilePicData):
    try:
        user_image_doc = collection_images.find_one(
            {"username": profile_pic_data.username})

        if user_image_doc:
            collection_images.update_one(
                {"_id": user_image_doc["_id"]},
                {"$set": {"profilePicture": profile_pic_data.imageURL}}
            )

            return {"message": "Profile picture uploaded successfully"}
        else:
            return {"message": "Username not found"}
    except Exception as e:
        print(f"Exception uploading profile picture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-user-data/{username}")
async def get_user_data(username: str):
    user_data = collection_user.find_one({"username": username})
    # print(user_data)
    if user_data:
        user_data['_id'] = str(user_data['_id'])
        return user_data
    else:
        return {"message": "User not found"}


@app.get("/get-user-images/{username}")
async def get_user_images(username: str):
    user_images_data = collection_images.find_one({"username": username})
    # print(user_images_data)
    if user_images_data:
        user_images_data['_id'] = str(user_images_data['_id'])
        return user_images_data
    else:
        return {"message": "User images not found"}


@app.get("/get-user-transactions/{username}")
async def get_user_transactions(username: str):
    user_transactions_data = collection_transactions.find_one(
        {"username": username})
    # print(user_transactions_data)
    if user_transactions_data:
        user_transactions_data['_id'] = str(user_transactions_data['_id'])
        return user_transactions_data
    else:
        return {"message": "User transactions not found"}


@app.get("/get-user-cards/{username}")
async def get_user_cards(username: str):
    user_cards_data = collection_cards.find_one(
        {"username": username})
    if user_cards_data:
        user_cards_data['_id'] = str(user_cards_data['_id'])
        return user_cards_data
    else:
        raise HTTPException(status_code=404, detail="Username not found")


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
    # print(username, password)
    is_valid = validate_user(username, password)
    # print("login", is_valid)
    if (is_valid):
        return True
    else:
        return False


@app.post("/generate-otp")
def generate_otp_endpoint(mobileNumber: otpData):
    # global temporary_otp
    otp = generate_otp()
    print(otp)

    try:
        print(mobileNumber)
        send_otp_via_sms(mobileNumber.mobileNumber, otp)

        return otp
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send OTP: {str(e)}")


class CardBase(BaseModel):
    username: str
    bankName: str
    cardNumber: str
    updatedIssueDate: str
    activeStatus: bool


@app.post("/store-card-details")
async def store_card_details(card_details: CardBase):

    if collection_cards.find_one({"username": card_details.username}):
        result = collection_cards.update_one(
            {"username": card_details.username},
            {"$set": card_details.dict(exclude={"username"})}
        )
        if result.modified_count > 0:
            return {"message": "Card details updated successfully"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to update card details")

    card_document = card_details.dict()

    result = collection_cards.insert_one(card_document)

    if result.inserted_id:
        return {"message": "Card details stored successfully", "card_id": str(result.inserted_id)}
    else:
        return {"message": "Failed to store card details"}


@app.post("/update-active-status/{cardNumber}")
async def update_active_status(cardNumber: str):
    user_data = collection_cards.find_one({"cardNumber": cardNumber})
    if user_data:
        current_status = user_data.get("activeStatus")
        updated_status = not current_status

        result = collection_cards.update_one(
            {"cardNumber": cardNumber},
            {"$set": {"activeStatus": updated_status}}
        )

        if result.modified_count > 0:
            return {"message": f"Active status updated to {updated_status} for {cardNumber}"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to update active status")
    else:
        raise HTTPException(status_code=404, detail="Card Number not found")


@app.post("/update-online-Status/{cardNumber}")
async def update_active_status(cardNumber: str):
    user_data = collection_cards.find_one({"cardNumber": cardNumber})
    if user_data:
        current_status = user_data.get("onlineStatus")
        updated_status = not current_status

        result = collection_cards.update_one(
            {"cardNumber": cardNumber},
            {"$set": {"onlineStatus": updated_status}}
        )

        if result.modified_count > 0:
            return {"message": f"Online Status updated to {updated_status} for {cardNumber}"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to update online status")
    else:
        raise HTTPException(status_code=404, detail="Card Number not found")


@app.post("/update-international-status/{cardNumber}")
async def update_active_status(cardNumber: str):
    user_data = collection_cards.find_one({"cardNumber": cardNumber})
    if user_data:
        current_status = user_data.get("intStatus")
        updated_status = not current_status

        result = collection_cards.update_one(
            {"cardNumber": cardNumber},
            {"$set": {"intStatus": updated_status}}
        )

        if result.modified_count > 0:
            return {"message": f"International status updated to {updated_status} for {cardNumber}"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to update active status")
    else:
        raise HTTPException(status_code=404, detail="Card Number not found")


@app.post("/set-default-card")
async def set_default_card(user_id: str, cardNumber: str, ):
    try:
        print(f"Received user_id: {user_id}, card_number: {cardNumber}")

        result = collection_cards.update_many(
            {"userId": user_id},
            {"$set": {"default": False}}
        )

        print(f"Updated {result.modified_count} documents to default=False")

        result = collection_cards.find_one_and_update(
            {"userId": user_id, "cardNumber": cardNumber},
            {"$set": {"default": True}}
        )

        if not result:
            raise HTTPException(
                status_code=404, detail="Card not found for the user")

        return {"message": f"Default card updated for user: {user_id}"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}")


class Transaction(BaseModel):
    time: datetime
    amount: int
    videoURL: str
    status: bool


class UserTransactions(BaseModel):
    username: str
    transactions: List[Transaction]


@app.post("/get-transactions")
async def get_transactions_in_date_range(username: str, start_date: str, end_date: str) -> List[Transaction]:
    try:
        start_date = parser.isoparse(start_date)
        end_date = parser.isoparse(end_date)

        query = {
            "username": username,
            "transaction.time": {
                "$gte": start_date,
                "$lt": end_date
            }
        }

        user_data = collection_transactions.find_one(query)

        if not user_data:
            raise HTTPException(
                status_code=404, detail="No transactions found for this date range")

        transactions = user_data.get("transaction", [])

        filtered_transactions = [
            Transaction(**transaction) for transaction in transactions
            if start_date <= transaction["time"] < end_date
        ]

        return filtered_transactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class updatedData(BaseModel):
    faceName: str
    updatedName: str
    username: str


@app.post("/update-face-name")
async def update_face_name(data: updatedData):
    try:
        query = {"username": data.username}
        user_data = collection_images.find_one(query)

        if not user_data:
            raise HTTPException(status_code=404, detail="Username not found")

        user_faces = user_data.get("user_faces", [])

        for face in user_faces:
            if face.get("faceName") == data.faceName:
                face["faceName"] = data.updatedName
                break

        collection_images.update_one(
            {"username": data.username},
            {"$set": {"user_faces": user_faces}}
        )

        return {"message": "Face name updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FaceNameDelete(BaseModel):
    username: str
    faceName: str


@app.post("/delete-face-name")
async def delete_face_name(data: FaceNameDelete):
    try:
        query = {"username": data.username}
        user_data = collection_images.find_one(query)

        if not user_data:
            raise HTTPException(status_code=404, detail="Username not found")

        user_faces = user_data.get("user_faces", [])

        filtered_faces = [face for face in user_faces if face.get(
            "faceName") != data.faceName]

        collection_images.update_one(
            {"username": data.username},
            {"$set": {"user_faces": filtered_faces}}
        )

        return {"message": f"Objects with faceName '{data.faceName}' deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CardInfo(BaseModel):
    userId: str
    bankName: str
    balance: float
    cardNumber: str
    issueDate: str
    pincode: str
    activeStatus: bool = True
    onlineStatus: bool = True
    intStatus: bool = True
    default: bool = False


@app.post("/store-card-info")
async def store_card_info(card_info: CardInfo):
    try:
        card_data = card_info.dict()

        collection_cards.insert_one(card_data)

        return {"message": "Card information stored successfully"}

    except Exception as e:
        return {"error": f"Failed to store card information: {str(e)}"}


@app.get("/get-cards/{user_id}")
async def get_cards_by_user_id(user_id: str):
    try:
        cards = collection_cards.find({"userId": user_id})
        card_list = [json.loads(json_util.dumps(card)) for card in cards]

        if not card_list:
            return {"message": "No cards found for this user"}

        return card_list

    except Exception as e:
        return {"error": f"Failed to fetch cards: {str(e)}"}


# SMTP_SERVER = 'smtp.gmail.com'
# SMTP_PORT = 587
# SENDER_EMAIL = 'regit973@gmail.com'
# PASSWORD = 'downloadregit'


# @app.post("/send_otp_emails")
# async def send_otp_emails(emails: list[str]):
#     otp_code = generate_otp()  # Generate OTP
#     sent_successfully = send_emails(emails, otp_code)  # Send OTP to emails
#     if sent_successfully:
#         return {"message": "OTP sent successfully"}
#     else:
#         return {"message": "Failed to send OTP"}


# def generate_otp():
#     return str(random.randint(1000, 9999))  # Generate a random 4-digit OTP


# def send_emails(emails, otp):
#     message = MIMEMultipart()
#     message['From'] = SENDER_EMAIL
#     message['Subject'] = 'Your OTP'

#     body = f'Your OTP is: {otp}'
#     message.attach(MIMEText(body, 'plain'))

#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(SENDER_EMAIL, PASSWORD)
#             for email in emails:
#                 message['To'] = email
#                 server.sendmail(SENDER_EMAIL, email, message.as_string())
#             server.quit()
#         return True  # Email(s) sent successfully
#     except Exception as e:
#         print(f"Email sending failed: {e}")
#         return False  # Failed to send emails


# twilio recovery code
# N9N6XJ3H1EWKX1XDKAJXPKEP
