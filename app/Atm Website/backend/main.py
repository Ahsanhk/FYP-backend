# from fastapi import Depends, FastAPI, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from pydantic import BaseModel
# from datetime import datetime, timedelta
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# import motor.motor_asyncio

# SECRET_KEY = "2261ad8242de1aefb80712940f669a5ff85ac2482e6ed6b9e31b122b589f8c21"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30

# # client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
# # database = client["test"]
# # collection = database["user"]

# db = {
#     "tim": {
#         "username": "tim",
#         "full_name": "Tim Ruscica",
#         "email": "tim@gmail.com",
#         "hashed_password": "$2b$12$HxWHkvMuL7WrZad6lcCfluNFj1/Zp63lvP5aUrKlSTYtoFzPXHOtu",
#         "disabled": False
#     }
# }


# class Token(BaseModel):
#     access_token: str
#     token_type: str


# class TokenData(BaseModel):
#     username: str or None = None


# class User(BaseModel):
#     username: str
#     email: str or None = None
#     full_name: str or None = None
#     disabled: bool or None = None


# class UserInDB(User):
#     hashed_password: str


# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# app = FastAPI()


# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)


# def get_password_hash(password):
#     return pwd_context.hash(password)


# async def get_user(username: str):
#     user = await collection.find_one({"username": username})
#     # print({"username": username})
#     print("user in get_user: ", user)
#     return user


# async def authenticate_user(username: str, password: str):
#     user = await get_user(username)
#     print('user in authentication', user)
#     if user or not verify_password(password, user["hashed_password"]):
#         return False
#     return user


# async def create_access_token(data: dict, expires_delta: timedelta or None = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)

#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt


# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
#                                          detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credential_exception

#         token_data = TokenData(username=username)
#     except JWTError:
#         raise credential_exception

#     user = get_user(username=token_data.username)
#     if user is None:
#         raise credential_exception

#     return user


# async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
#     if current_user.disabled:
#         raise HTTPException(status_code=400, detail="Inactive user")

#     return current_user


# @app.post("/token", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = await authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
#                             detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires)
#     return {"access_token": access_token, "token_type": "bearer"}


# @app.get("/users/me/", response_model=User)
# async def read_users_me(current_user: User = Depends(get_current_active_user)):
#     return current_user


# @app.get("/users/me/items")
# async def read_own_items(current_user: User = Depends(get_current_active_user)):
#     return [{"item_id": 1, "owner": current_user}]


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

from test3 import generate_frames
from pincodeValidation import validate_pincode
from usernameAuth import get_user_by_username
from multipleFaceCheck import detect_faces
from emotionDetection import detect_emotion
from faceCoverCheck import isFaceCovered
from faceRecognition import fetch_image_url_from_db, decode_image_from_url, generate_frames, recognize_face_in_frame

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

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load YOLO
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# check username
@app.get("/get_user/{username}")
async def get_user_by_username_route(username: str):
    user_data = get_user_by_username(username)
    # return {"user_data": user_data}
    if username is None:
        return {"isAvailable": False}
    else:
        return {"isAvailable": True}


# fetch user data
@app.get("/get-user-data/{username}")
async def get_user_data(username: str):
    user_data = collection_user.find_one({"username": username})
    print(user_data)
    if user_data:
        user_data['_id'] = str(user_data['_id'])
        return user_data
    else:
        return {"message": "User not found"}


# validate pincode
@app.get("/validate_pincode")
async def validate_pincode_route(username: str, pincode: str):
    is_valid = validate_pincode(username, pincode)
    print(is_valid)
    return {"isValid": is_valid}


# global videoURL

stop_processing = False
cloudinary_url = None


class TransactionData(BaseModel):
    # time: str
    amount: float
    username: str
    videoURL: str
    status: bool = False


@app.post("/upload-transaction-data")
async def upload_transaction_data(transaction_data: TransactionData):
    global stop_processing
    stop_processing = True
    # global cloudinary_url

    try:
        user_transaction_doc = collection_transactions.find_one(
            {"username": transaction_data.username})

        if user_transaction_doc:
            new_transaction = {
                "time": datetime.now(),
                "amount": transaction_data.amount,
                "videoURL": transaction_data.videoURL,
                "status": True
            }

            user_transaction_doc["transaction"].append(new_transaction)

            collection_transactions.update_one(
                {"_id": user_transaction_doc["_id"]},
                {"$set": {"transaction": user_transaction_doc["transaction"]}}
            )

            return {"message": "Transaction data uploaded successfully"}
        else:
            return {"message": "Username not found"}
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


def draw_bounding_boxes(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def check_multiple_faces():
    # global videoURL
    video = cv2.VideoCapture(0)
    out = cv2.VideoWriter(
        'output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
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
    out.release()
    cv2.destroyAllWindows()
    return False

    compression_command = 'ffmpeg -i output.avi -vcodec libx264 -crf 24 output_compressed.mp4'
    subprocess.run(compression_command, shell=True)

    cloudinary.config(
        cloud_name="dlk8tzdu3",
        api_key="994623638344852",
        api_secret="YGFyNfSaEavUMv_3MV6xDbOLzsI",
    )

    upload_result = cloudinary.uploader.upload(
        "output_compressed.mp4", resource_type='video', folder='user_videos')

    videoURL = upload_result['url']
    print(videoURL)


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

        mouth_covered = not np.all(mouth_landmarks)
        nose_covered = not np.all(nose_landmarks)
        jaw_covered = not np.all(jaw_landmarks)
        print('mouth', mouth_covered)
        print('nose', nose_covered)
        print('jaw', jaw_covered)

        if nose_covered or jaw_covered:
            print('one of the mouse, nose or jaw is covered')
            faces.append((x, y, w, h, True))
            # return True
        else:
            # return False
            faces.append((x, y, w, h, False))

    return faces


@app.get("/multiple_face_detection")
async def multiple_face_detection_route():
    is_multiple_faces = check_multiple_faces()
    print(is_multiple_faces)
    return {"multiple_faces_detected": is_multiple_faces}


@app.get("/face_cover_check")
async def check_face_cover_route():
    is_face_covered = check_face_cover()
    return {"face covered/multiple faces": is_face_covered}


async def emotion_detection():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getUnconnectedOutLayersNames()

    global cloudinary_url
    global stop_processing

    emotion_model = DeepFace.build_model('Emotion')
    cap = cv2.VideoCapture(0)

    cloudinary.config(
        cloud_name="dlk8tzdu3",
        api_key="994623638344852",
        api_secret="YGFyNfSaEavUMv_3MV6xDbOLzsI",
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rec = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while not stop_processing:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Error: Could not read frame.")
            break

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # if len(boxes) > 1:
        #     return ("multiple faces" : True)
        if len(boxes) > 0:
            x, y, w, h = boxes[0]
            face = frame[y:y+h, x:x+w]

            if face.size > 0:
                emotion_results = DeepFace.analyze(
                    face, actions=['emotion'], enforce_detection=False)

                if emotion_results and 'dominant_emotion' in emotion_results[0]:
                    dominant_emotion = emotion_results[0]['dominant_emotion']
                    emotion_accuracy = emotion_results[0]['emotion'][dominant_emotion]
                    cv2.putText(frame, f"Emotion: {dominant_emotion} ({emotion_accuracy:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if emotion_accuracy > 0.6 and dominant_emotion.lower() == 'fear':
                        stop_processing = True
                        print("Detected 'fear'. Stopping further detection.")
                        # return True

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rec.write(frame)

        if stop_processing:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    rec.release()
    cv2.destroyAllWindows()

    upload_result = cl_upload("output.avi",
                              resource_type='video', folder='user_videos')

    cloudinary_url = upload_result['secure_url']
    print("Uploaded video URL:", cloudinary_url)

    if stop_processing:
        return True
    else:
        return False


@app.get("/emotion-detection")
async def video_feed():
    global stop_processing
    stop_processing = False
    # event = asyncio.Event()
    emotion_result = await emotion_detection()
    print('emotion result: ', emotion_result)
    if emotion_result:
        return True
    else:
        return False


def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


async def face_detection(username):
    cap = cv2.VideoCapture(0)

    cloudinary.config(
        cloud_name="dlk8tzdu3",
        api_key="994623638344852",
        api_secret="YGFyNfSaEavUMv_3MV6xDbOLzsI",
    )

    user_data = collection_images.find_one({"username": username})
    # reference_image_url = "https://res.cloudinary.com/dlk8tzdu3/image/upload/v1700467398/test_preset/qluimaiuagyv31nrvwaj.jpg"

    if user_data and "user_faces" in user_data:
        first_face = user_data["user_faces"][0]
        reference_image_url = first_face.get("imageUrl")

        reference_image = load_image_from_url(reference_image_url)
        reference_encoding = face_recognition.face_encodings(reference_image)[
            0]

        duration = 2
        start_time = time.time()

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()

            timeout = 2
            face_start_time = time.time()

            while (time.time() - face_start_time) < timeout:
                face_locations = face_recognition.face_locations(frame)

                if face_locations:
                    face_encodings = face_recognition.face_encodings(
                        frame, face_locations)

                    for face_encoding in face_encodings:
                        results = face_recognition.compare_faces(
                            [reference_encoding], face_encoding)

                        if results[0]:
                            cap.release()
                            cv2.destroyAllWindows()
                            print("Match found!")
                            return True

                break

            if (time.time() - start_time) >= duration:
                cv2.imwrite("frame.jpg", frame)
                upload_result = cloudinary.uploader.upload(
                    "frame.jpg", folder="alert_images")
                imgURL = upload_result.get('url')
                print(imgURL)
                print("No match found. Frame uploaded to Cloudinary.")

                TWILIO_ACCOUNT_SID = "AC8d12dac8399326b3699e757fec3a51ee"
                TWILIO_AUTH_TOKEN = "2b2ab2d3972fe87a8b98e778efda33b3"
                DESTINATION_PHONE_NUMBER = "+923337010789"

                client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

                message = client.messages.create(
                    from_="+13343779670",
                    body=f"Alert: A transaction was made by an unknown person. Click the link to see the person trying to use your card {imgURL}.",
                    to=DESTINATION_PHONE_NUMBER
                )
                print("text sent")

                break

    cap.release()
    cv2.destroyAllWindows()
    print("No match found.")
    return False

    TWILIO_ACCOUNT_SID = os.getenv("AC8d12dac8399326b3699e757fec3a51ee")
    TWILIO_AUTH_TOKEN = os.getenv("2b2ab2d3972fe87a8b98e778efda33b3")
    # TWILIO_PHONE_NUMBER = os.getenv("+19386661883")
    MONGODB_CONNECTION_STRING = os.getenv("mongodb://localhost:27017")


@app.get("/face-recognition/{username}", response_class=JSONResponse)
async def face_detection_route(request: Request, username: str):
    isFaceRecognized = await face_detection(username)
    return JSONResponse(content=isFaceRecognized)


# def draw_bounding_boxes_cover(frame, faces):
#     for face in faces:
#         x, y, w, h, covered = face
#         # Green if not covered, red if covered
#         color = (0, 255, 0) if not covered else (0, 0, 255)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

#         if covered:
#             face_region = frame[y:y+h, x:x+w]

#             # Detect facial landmarks within the bounding box
#             face_rect = dlib.rectangle(
#                 left=x, top=y, right=x + w, bottom=y + h)
#             shape = predictor(frame, face_rect)
#             shape = face_utils.shape_to_np(shape)

#             # Define the eye landmarks
#             left_eye = shape[36:42]
#             right_eye = shape[42:48]

#             # Check if the eyes are detected and draw rectangles if covered
#             if not np.all(left_eye):
#                 left_eye_x = np.min(left_eye[:, 0])
#                 left_eye_y = np.min(left_eye[:, 1])
#                 left_eye_w = np.max(left_eye[:, 0]) - left_eye_x
#                 left_eye_h = np.max(left_eye[:, 1]) - left_eye_y
#                 cv2.rectangle(face_region, (left_eye_x, left_eye_y), (left_eye_x +
#                               left_eye_w, left_eye_y + left_eye_h), (0, 0, 255), 2)

#             if not np.all(right_eye):
#                 right_eye_x = np.min(right_eye[:, 0])
#                 right_eye_y = np.min(right_eye[:, 1])
#                 right_eye_w = np.max(right_eye[:, 0]) - right_eye_x
#                 right_eye_h = np.max(right_eye[:, 1]) - right_eye_y
#                 cv2.rectangle(face_region, (right_eye_x, right_eye_y), (right_eye_x +
#                               right_eye_w, right_eye_y + right_eye_h), (0, 0, 255), 2)
