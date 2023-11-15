from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from twilio.rest import Client
from pymongo import MongoClient
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class otpData(BaseModel):
    mobileNumber: str


TWILIO_ACCOUNT_SID = os.getenv("AC10878b2163268a12a02b190255fc9e38")
TWILIO_AUTH_TOKEN = os.getenv("aacedc854bc74414247e08d4d404582f")
# TWILIO_PHONE_NUMBER = os.getenv("+19386661883")
MONGODB_CONNECTION_STRING = os.getenv("mongodb://localhost:27017")

# print(os.getenv("TWILIO_ACCOUNT_SID"))
# print(os.getenv("TWILIO_AUTH_TOKEN"))
# print(os.getenv("TWILIO_PHONE_NUMBER"))

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def generate_otp():
    # Generate a 6-digit random OTP
    return str(random.randint(100000, 999999))


def send_otp_via_sms(to, otp):
    message = client.messages.create(
        from_="+19386661883",
        body=f"Your OTP is: {otp}",
        to=to
    )
    return message
