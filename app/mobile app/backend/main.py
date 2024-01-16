# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from typing import Any
from typing import Dict
import hashlib
import sched
import bcrypt
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
    username: str


class ProfilePicData(BaseModel):
    imageURL: str
    username: str


class UploadUserFaceRequest(BaseModel):
    userId: str
    imageUrl: str
    faceName: str


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


@app.get("/check-username/{username}")
async def check_username(username: str):
    existing_user = collection_user.find_one({"username": username})

    if existing_user:
        return {"message": "Username already taken", "isTaken": True}
    else:
        return {"isTaken": False}


class UploadUserFaceRequest(BaseModel):
    user_id: str
    imageUrl: str
    faceName: str


@app.post("/upload-user-face")
async def upload_user_face(request_data: UploadUserFaceRequest):
    try:
        userId = request_data.user_id
        imageUrl = request_data.imageUrl
        faceName = request_data.faceName

        face_data = {
            "userId": userId,
            "imageUrl": imageUrl,
            "faceName": faceName
        }

        collection_images.insert_one(face_data)

        return {"message": f"Image URL added for {userId}"}
    except Exception as e:
        print(f"Exception uploading user face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-user-images/{username}")
async def get_user_images(username: str):
    user_images_data = collection_images.find_one({"username": username})
    # print(user_images_data)
    if user_images_data:
        user_images_data['_id'] = str(user_images_data['_id'])
        return user_images_data
    else:
        return {"message": "User images not found"}


@app.get("/get-user-transactions/{card_id}")
async def get_user_transactions(card_id: str):
    try:
        transactions = list(collection_transactions.find({"card_id": card_id}))
        for transaction in transactions:
            transaction['_id'] = str(transaction['_id'])
        return transactions
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching user transactions: {str(e)}")


@app.get("/get-user-cards/{username}")
async def get_user_cards(username: str):
    user_cards_data = collection_cards.find_one(
        {"username": username})
    if user_cards_data:
        user_cards_data['_id'] = str(user_cards_data['_id'])
        return user_cards_data
    else:
        raise HTTPException(status_code=404, detail="Username not found")


@app.get("/get-default-card/{user_id}")
async def get_default_card(user_id: str):
    try:
        cards = collection_cards.find({"userId": user_id})

        default_card = next(
            (card for card in cards if card.get("default")), None)

        if default_card:
            # Convert ObjectId to string
            card_id = str(default_card["_id"])

            transactions = list(collection_transactions.find(
                {"card_id": card_id}))
            # transactions = list(transactions_cursor)
            # print(transactions)

            transactions = [
                {**transaction, "_id": str(transaction["_id"])}
                for transaction in transactions
            ]
            return {"transactions": transactions}
        else:
            raise HTTPException(
                status_code=404, detail="Default card not found for the user")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/authentication")
async def validate_user_route(username: str, password: str):
    is_valid = validate_user(username, password)
    if (is_valid):
        return True
    else:
        return False


def delete_expired_otps(mobile_number):
    collection_otp.delete_many({"mobileNumber": mobile_number})


@app.post("/generate-otp")
def generate_otp_endpoint(mobileNumber: otpData):
    otp = generate_otp()
    print(otp)
    hashed_otp = hashlib.sha256(otp.encode()).hexdigest()

    try:
        otp_data = {
            "mobileNumber": mobileNumber.mobileNumber,
            "otp": hashed_otp,
        }

        result = collection_otp.insert_one(otp_data)

        # send_otp_via_sms(mobileNumber.mobileNumber, otp)

        return {"otpId": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send OTP: {str(e)}")


class otpValData(BaseModel):
    otpId: str
    otp: str


def hash_otp(otp):
    return hashlib.sha256(otp.encode()).hexdigest()


@app.post('/validate-otp')
async def validate_otp(payload: otpValData):
    try:
        otp_id = ObjectId(payload.otpId)
        otp_data = collection_otp.find_one({"_id": otp_id})

        if otp_data:
            hashed_input_otp = hash_otp(payload.otp)

            if hashed_input_otp == otp_data["otp"]:
                return JSONResponse(content={"isMatched": True})
            else:
                return JSONResponse(content={"isMatched": False})
        else:
            raise HTTPException(status_code=404, detail="OTP not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error validating OTP: {str(e)}")


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


@app.get("/get-user-faces/{user_id}")
async def get_user_images(user_id: str):
    user_images_data = list(collection_images.find({"userId": user_id}))

    if user_images_data:
        for image_data in user_images_data:
            image_data['_id'] = str(image_data['_id'])
        return user_images_data
    else:
        return {"message": "User images not found"}


class updatedData(BaseModel):
    face_id: str
    updatedName: str


@app.post("/update-face-name")
async def update_face_name(data: updatedData):
    try:
        query = {"_id": ObjectId(data.face_id)}
        user_data = collection_images.find_one(query)

        if not user_data:
            raise HTTPException(status_code=404, detail="Face ID not found")

        collection_images.update_one(
            {"_id": ObjectId(data.face_id)},
            {"$set": {"faceName": data.updatedName}}
        )

        return {"message": "Face name updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FaceDelete(BaseModel):
    face_id: str


@app.post("/delete-face-name")
async def delete_face(data: FaceDelete):
    try:
        query = {"_id": ObjectId(data.face_id)}
        face_data = collection_images.find_one(query)

        if not face_data:
            raise HTTPException(status_code=404, detail="Face ID not found")

        collection_images.delete_one(query)

        return {"message": f"Face with ID '{data.face_id}' deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CardData(BaseModel):
    bankName: str
    cardNumber: str
    issuedate: str
    pincode: str
    userId: str


@app.post("/store-card-info")
async def store_card_info(card_info: CardData):
    try:
        print("card info: ", card_info)
        existing_card_count = collection_cards.count_documents(
            {"userId": card_info.userId})
        is_default = existing_card_count == 0

        balance = round(random.uniform(10000, 1000000), 2)

        hashed_pin = None

        try:
            hashed_pin = bcrypt.hashpw(
                card_info.pincode.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")
        except Exception as e:
            print("Hashing Exception:", str(e))

        card_data = card_info.dict()

        card_data.update(
            {
                "bankName": card_info.bankName,
                "cardNumber": card_info.cardNumber,
                "userId": card_info.userId,
                "issuedate": card_info.issuedate,
                "balance": balance,
                "activeStatus": True,
                "onlineStatus": True,
                "intStatus": True,
                "default": is_default,
                "pincode": hashed_pin,
                "assignedFaces": [],
            }
        )
        print(card_data)

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


@app.delete("/delete-card/{card_id}")
async def delete_card(card_id: str):
    try:
        query = {"_id": ObjectId(card_id)}
        card_data = collection_cards.find_one(query)

        if not card_data:
            raise HTTPException(status_code=404, detail="Card ID not found")

        collection_cards.delete_one(query)

        return {"message": f"Card with ID '{card_id}' deleted successfully"}
    except errors.PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))


class AssignedFace(BaseModel):
    faceId: str


class UpdateCardFacesRequest(BaseModel):
    card_id: str
    selectedFaces: List[AssignedFace]


@app.post("/update-card-faces")
async def update_card_faces(req: UpdateCardFacesRequest):
    try:
        card_id = req.card_id
        assigned_faces = req.selectedFaces

        card = collection_cards.find_one({"_id": ObjectId(card_id)})

        if card is None:
            raise HTTPException(status_code=404, detail="Card ID not found")

        updated_faces = [
            {"face_id": face.faceId}
            for face in assigned_faces
        ]

        collection_cards.update_one(
            {"_id": ObjectId(card_id)},
            {"$set": {"assignedFaces": updated_faces}}
        )

        return {"message": f"Assigned faces updated for card {card_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AssignedFacesRequest(BaseModel):
    assignedFaces: List[dict]


@app.post("/extract-face-names")
async def extract_face_names(request: AssignedFacesRequest):
    try:
        assigned_faces = request.assignedFaces
        face_ids = [ObjectId(face["face_id"]) for face in assigned_faces]

        matching_images = collection_images.find({"_id": {"$in": face_ids}})
        face_names = list(set([img["faceName"]
                          for img in matching_images if "faceName" in img]))

        return {"faceNames": face_names}

    except Exception as e:
        return {"error": str(e)}


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
