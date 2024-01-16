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
    # username: str
    mobileNumber: str


TWILIO_ACCOUNT_SID = os.getenv("AC8d12dac8399326b3699e757fec3a51ee")
TWILIO_AUTH_TOKEN = os.getenv("2b2ab2d3972fe87a8b98e778efda33b3")
# TWILIO_PHONE_NUMBER = os.getenv("+19386661883")
MONGODB_CONNECTION_STRING = os.getenv("mongodb://localhost:27017")

# print(os.getenv("TWILIO_ACCOUNT_SID"))
# print(os.getenv("TWILIO_AUTH_TOKEN"))
# print(os.getenv("TWILIO_PHONE_NUMBER"))

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def generate_otp():
    return str(random.randint(100000, 999999))


def send_otp_via_sms(to, otp):
    print(to)
    print(otp)
    message = client.messages.create(
        from_="+13343779670",
        body=f"Your OTP is: {otp}",
        to=to
    )
    return message
