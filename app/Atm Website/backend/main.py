import bcrypt
import io
import pyautogui
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.websockets import WebSocketDisconnect
from fastapi import FastAPI, WebSocket
from bson import ObjectId
import threading
import mediapipe as mp
from typing import Optional
import asyncio
from dateutil import parser
from dateutil import parser as date_parser
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
import os
from fastapi import FastAPI
from twilio.rest import Client
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from fastapi import FastAPI, UploadFile, File, Request, Response, HTTPException, status, Depends, Query
import cv2
import numpy as np
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display, clear_output
import time
import dlib
import subprocess
from imutils import face_utils
import cloudinary.uploader
from cloudinary.uploader import upload as cl_upload
from jose import JWTError, jwt
from datetime import datetime, timedelta
from deepface import DeepFace
import json
from pymongo import MongoClient
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# connecion with MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection_user = db["user"]
collection_transactions = db["user_transactions"]
collection_images = db["user_images"]
collection_cards = db["user_cards"]
collection_dump = db["dump_transactions"]

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Loading YOLO
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

emotion_detection_task = None


@app.get('/get-user/{cardNumber}')
async def get_user(cardNumber: str):
    user_data = collection_cards.find_one({"cardNumber": cardNumber})

    if user_data:
        return {
            "exists": True,
            "userId": user_data["userId"],
            "activeStatus": user_data.get("activeStatus", None)
        }
    else:
        return {"exists": False}


class CardDetails(BaseModel):
    cardNumber: str
    pincode: str


@app.post('/validate-pincode')
async def validate_pin(card: CardDetails):
    card_data = collection_cards.find_one({"cardNumber": card.cardNumber})

    if card_data:
        hashed_pin = card_data.get("pincode")
        entered_pin = card.pincode.encode("utf-8")

        if bcrypt.checkpw(entered_pin, hashed_pin.encode("utf-8")):
            return {"isValid": True, "userId": card_data.get("userId")}

    return {"isValid": False}


@app.get("/get-card-details/{card_number}")
async def get_card_details(card_number: str):
    card_data = collection_cards.find_one({"cardNumber": card_number})

    if card_data:
        card_data["_id"] = str(card_data["_id"])

        return card_data


# fetch user data


@app.get("/get-user-data/{user_id}")
async def get_user_data(user_id: str):
    try:
        user_object_id = ObjectId(user_id)
        user_data = collection_user.find_one({"_id": user_object_id})

        if user_data:
            user_data['_id'] = str(user_data['_id'])
            return user_data
        else:
            return {"message": "User not found"}
    except Exception as e:
        return {"message": "Invalid user_id format"}


cloudinary_url = None


class TransactionData(BaseModel):
    card_id: str
    amount: float
    videoUrl: str
    # status: bool = True


@app.post("/upload-transaction-data")
async def upload_transaction_data(transaction_data: TransactionData):

    try:
        new_transaction = {
            "card_id": transaction_data.card_id,
            "time": datetime.now(),
            "amount": transaction_data.amount,
            "videoURL": transaction_data.videoUrl,
            "status": True
        }

        collection_transactions.insert_one(new_transaction)

        return {"message": "Transaction data uploaded successfully"}
    except Exception as e:
        print(f"Exception uploading transaction data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def detect_multiple_faces(frame):
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    faces = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i.item()
        x, y, w, h = boxes[i]
        faces.append((x, y, w, h))

    return faces


def check_covered_face(frame):
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    faces = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i.item()
        x, y, w, h = boxes[i]
        faces.append((x, y, w, h))

    return faces


def draw_bounding_boxes(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def check_multiple_faces():
    video = cv2.VideoCapture(0)

    multiple_faces = False
    stop_detection = False

    duration = 2
    start_time = time.time()

    while (time.time() - start_time) < duration:
        ret, frame = video.read()
        if ret:
            faces = detect_multiple_faces(frame)
            out.write(frame)
            draw_bounding_boxes(frame, faces)
            cv2.imshow('Checking multiple faces', frame)
            if len(faces) > 1:
                multiple_faces = True
                return True
                break
            if multiple_faces:
                return True
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()
    return False


def check_face_cover():
    video = cv2.VideoCapture(0)
    isFaceCovered = False
    # isMultipleFaces = False

    duration = 2
    start_time = time.time()

    while (time.time() - start_time) < duration:
        ret, frame = video.read()
        if ret:
            faces = detect_facial_landmarks(frame)
            print('faces:', faces)
            # draw_bounding_boxes(frame, faces)
            if len(faces) == 0:
                return True
            if len(faces) > 1:
                isFaceCovered = True
                return isFaceCovered
            if any(face[-1] for face in faces):
                isFaceCovered = True
                return isFaceCovered
            cv2.imshow('Video', frame)
            # if len(faces) > 1:
            #     multiple_faces_bool = True
            #     return True

            if isFaceCovered:
                return True
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()
    return False


def detect_facial_landmarks(frame):
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    faces = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    for i in indices:
        index = i.item()
        x, y, w, h = boxes[index]

        face_rect = dlib.rectangle(
            left=x, top=y, right=x + w, bottom=y + h)

        shape = predictor(frame, face_rect)
        shape = face_utils.shape_to_np(shape)

        mouth_landmarks = shape[48:68]
        nose_landmarks = shape[27:36]
        jaw_landmarks = shape[0:17]

        mouth_covered = np.any(mouth_landmarks)
        nose_covered = np.any(nose_landmarks)
        jaw_covered = np.any(jaw_landmarks)
        print('mouth', not mouth_covered)
        print('nose', not nose_covered)
        print('jaw', not jaw_covered)

        if not (mouth_covered or nose_covered or jaw_covered):
            # print('none of the mouth, nose or jaw is covered')
            faces.append((x, y, w, h, False))
        else:
            # print('one of the mouth, nose or jaw is covered')
            faces.append((x, y, w, h, True))

    # return faces


@app.get("/multiple_face_detection")
async def multiple_face_detection_route():
    is_multiple_faces = check_multiple_faces()
    print(is_multiple_faces)
    return {"multiple_faces_detected": is_multiple_faces}


@app.get("/face_cover_check")
async def check_face_cover_route():
    is_face_covered = check_face_cover()
    return {"face covered/multiple faces": is_face_covered}


async def check_multiple_faces_with_emotion():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getUnconnectedOutLayersNames()

    emotion_model = DeepFace.build_model('Emotion')
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    processing_time = 2

    while (time.time() - start_time) < processing_time:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Error: Could not read frame.")
            break

        # Emotion detection phase
        faces = detect_multiple_faces(frame)

        if len(faces) == 0:
            print("No faces detected.")
            return {"message": "No faces detected.", "shouldStop": True}
        elif len(faces) > 1:
            print("Multiple faces detected.")
            return {"message": "Multiple faces detected.", "shouldStop": True}
        elif len(faces) == 1:
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]

            if face.size > 0:
                emotion_results = DeepFace.analyze(
                    face, actions=['emotion'], enforce_detection=False)

                if emotion_results and 'dominant_emotion' in emotion_results[0]:
                    dominant_emotion = emotion_results[0]['dominant_emotion']
                    emotion_accuracy = emotion_results[0]['emotion'][dominant_emotion]
                    cv2.putText(frame, f"Emotion: {dominant_emotion} ({emotion_accuracy:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if dominant_emotion.lower() == 'fear' and emotion_accuracy > 0.6:
                        print(
                            "Detected 'fear' in a single face. Stopping further detection.")
                        return {"message": "Detected 'fear' in a single face.", "shouldStop": True}

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if check_covered_face(frame):
            # print(
            #     'One of the mouth, nose, or jaw is covered. Stopping further detection.')
            return {"message": "One of the mouth, nose, or jaw is covered.", "shouldStop": True}

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return False


@app.get("/emotion-detection")
async def video_feed():
    event = asyncio.Event()
    emotion_result = await check_multiple_faces_with_emotion()
    print('emotion result: ', emotion_result)
    if emotion_result:
        return True
    else:
        return False


async def get_mobile_number(card_number: str):
    cards_data = collection_cards.find_one({"cardNumber": card_number})
    user_id = cards_data.get("userId") if cards_data else None

    if user_id:
        user_id_obj = ObjectId(user_id)

        user_data = collection_user.find_one({"_id": user_id_obj})
        mobile_number = user_data.get("mobileNumber") if user_data else None

        return mobile_number

    else:
        return {"error": "User ID not found for the given card number."}


def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


async def face_detection(cardNumber):
    cap = cv2.VideoCapture(0)

    cloudinary.config(
        cloud_name="dlk8tzdu3",
        api_key="994623638344852",
        api_secret="YGFyNfSaEavUMv_3MV6xDbOLzsI",
    )

    card_instance = collection_cards.find_one({"cardNumber": cardNumber})

    if not card_instance:
        raise HTTPException(
            status_code=404, detail=f"Card with cardNumber {cardNumber} not found")

    assigned_faces = card_instance.get("assignedFaces", [])
    print(assigned_faces)

    user_data = []

    for assigned_face in assigned_faces:
        face_id_str = assigned_face.get("face_id", "")

        try:
            face_id = ObjectId(face_id_str)
            face_instance = collection_images.find_one(
                {"_id": face_id}, {"imageUrl": 1})

            if face_instance and "imageUrl" in face_instance:
                user_data.append(face_instance["imageUrl"])
        except Exception as e:
            print(f"Error converting face_id to ObjectId: {e}")

    if user_data:
        reference_encodings = []

        for reference_image_url in user_data:

            reference_image = load_image_from_url(reference_image_url)

            if reference_image is not None:
                reference_encoding = face_recognition.face_encodings(reference_image)[
                    0]
                reference_encodings.append(
                    (reference_encoding, reference_image_url))

        duration = 2
        start_time = time.time()

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()

            face_locations = face_recognition.face_locations(frame)

            if face_locations:
                face_encodings = face_recognition.face_encodings(
                    frame, face_locations)

                for face_encoding in face_encodings:
                    for reference_encoding, matching_url in reference_encodings:
                        results = face_recognition.compare_faces(
                            [reference_encoding], face_encoding)

                        if any(results):
                            cap.release()
                            cv2.destroyAllWindows()
                            print(f"Match found with URL: {matching_url}")
                            return True

                # break

            if (time.time() - start_time) >= duration:
                cv2.imwrite("frame.jpg", frame)
                upload_result = cloudinary.uploader.upload(
                    "frame.jpg", folder="alert_images")
                imgURL = upload_result.get('url')
                # print(imgURL)
                # print("No match found. Frame uploaded to Cloudinary.")

                destination_phone_number = await get_mobile_number(cardNumber)
                # print(destination_phone_number)

                TWILIO_ACCOUNT_SID = "AC10878b2163268a12a02b190255fc9e38"
                TWILIO_AUTH_TOKEN = "aacedc854bc74414247e08d4d404582f"
                # destination_phone_number = '+923061668634'
                client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

                message = client.messages.create(
                    from_="+19386661883",
                    body=f"Alert: An unknown person tried to access your account. Click the link to see the person trying to use your card {imgURL}.",
                    to=destination_phone_number
                )
                # print("text sent")

                break

    cap.release()
    cv2.destroyAllWindows()
    print("No match found.")
    return False


@app.get("/face-recognition/{cardNumber}", response_class=JSONResponse)
async def face_detection_route(request: Request, cardNumber: str):
    isFaceRecognized = await face_detection(cardNumber)
    return JSONResponse(content=isFaceRecognized)


lock = threading.Lock()
frame = None
hands = mp.solutions.hands.Hands()

active_tasks = {}

cloudinary.config(
    cloud_name="dlk8tzdu3",
    api_key="994623638344852",
    api_secret="YGFyNfSaEavUMv_3MV6xDbOLzsI",
)


async def gesture_recognition(websocket: WebSocket, card_id: str):
    cap = cv2.VideoCapture(0)
    palm_cascade = cv2.CascadeClassifier('lpalm.xml')
    fist_cascade = cv2.CascadeClassifier('fist.xml')

    fist_detected = False
    recording_frames = []
    consecutive_fist_count = 0
    frame_count = 0
    new_transaction = {}

    try:
        recording = False
        last_frame = None
        frame = None

        while active_tasks.get(websocket, False):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            palms = palm_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            fist = fist_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            x, y, w, h = 0, 0, 0, 0

            for (x, y, w, h) in palms:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if len(fist) > 0:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                consecutive_fist_count += 1
                if consecutive_fist_count == 6:
                    fist_detected = True
                    await websocket.send_text(json.dumps({"type": "fist_recognized", "data": {"recognized": True}}))
                    # print("before break")
                    break
            else:
                consecutive_fist_count = 0

            if active_tasks.get(websocket, False) and frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
                recording_frames.append(frame)
                last_frame = frame
                if not recording:
                    print("Recording started")
                    recording = True
                cv2.imshow('Video Feed', frame)
                cv2.waitKey(1)

            frame_count += 1

            await asyncio.sleep(0.1)

        if fist_detected:
            print("Recording stopped on detection")
            video_path = 'recorded_video.mkv'
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(
                *'DIVX'), 30, (last_frame.shape[1], last_frame.shape[0]))
            for frame in recording_frames:
                out.write(frame)
            out.release()

            upload_result = cl_upload(
                video_path, resource_type='video', folder='user_videos')
            video_url = upload_result.get("secure_url", "")
            print("sending video url: ", video_url)
            await websocket.send_text(json.dumps({"type": "video_url", "data": {"url": video_url}}))

            new_transaction = {
                "card_id": card_id,
                "time": datetime.now(),
                "amount": "N/A",
                "videoURL": video_url,
                "status": False
            }

        collection_transactions.insert_one(new_transaction)

        cap.release()
        cv2.destroyAllWindows()

        # print(len(recording_frames))
        if recording_frames:
            print("Recording stopped")
            video_path = 'recorded_video.mkv'
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(
                *'DIVX'), 30, (last_frame.shape[1], last_frame.shape[0]))
            for frame in recording_frames:
                out.write(frame)
            out.release()

            upload_result = cl_upload(
                video_path, resource_type='video', folder='user_videos')
            video_url = upload_result.get("secure_url", "")
            print("sending video url: ", video_url)
            await websocket.send_text(json.dumps({"type": "video_url", "data": {"url": video_url}}))
        else:
            return {"fist_detected": fist_detected, "video_url in else": ""}

    except WebSocketDisconnect:
        cap.release()
        cv2.destroyAllWindows()
        active_tasks.pop(websocket, None)
        return {"fist_detected": False, "video_url in exception": ""}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start_task"):
                card_id = data.split(" ")[1]
                active_tasks[websocket] = True
                asyncio.create_task(gesture_recognition(websocket, card_id))
            elif data == "stop_task":
                active_tasks[websocket] = False
    except WebSocketDisconnect:
        active_tasks.pop(websocket, None)
