from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import base64
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = FastAPI()


async def fetch_image_url_from_db(username: str):
    # Assume you have a MongoDB connection already established
    # Replace the connection string and database/collection names with your own
    client = AsyncIOMotorClient("mongodb://localhost:27017/")
    db = client["test"]
    collection = db["user faces"]

    # Fetch the image URL from MongoDB based on the provided username
    result = await collection.find_one({"username": username})
    if result:
        return result.get("imageUrl")
    else:
        raise HTTPException(status_code=404, detail="User not found")


def decode_image_from_url(image_url: str):
    if image_url is None:
        raise HTTPException(status_code=404, detail="Image URL not found")

    # Check if the URL starts with 'data:image'
    if image_url.startswith('data:image'):
        # Assume the image is base64-encoded
        try:
            _, encoded_image = image_url.split(",", 1)
        except ValueError as ve:
            print("Error splitting image URL:", ve)
            raise HTTPException(
                status_code=400, detail="Invalid image URL format")
    else:
        # Assume the image is a regular URL, download it
        import requests
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail="Failed to download image from URL")

        encoded_image = base64.b64encode(response.content).decode()

    # Continue with the rest of the decoding process
    decoded_image = base64.b64decode(encoded_image)

    # Convert to NumPy array for further processing
    np_array = np.frombuffer(decoded_image, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def generate_frames(reference_image):
    # Open the default camera (you may need to change the camera index based on your setup)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Perform face recognition in the live frame
        face_recognition_result = recognize_face_in_frame(
            frame, reference_image)

        # Add a text label to the frame based on face recognition result
        label = "Face Recognized" if face_recognition_result else "Face Not Recognized"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as bytes for streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the camera when the generator ends
    cap.release()


def recognize_face_in_frame(frame, reference_image):
    # Convert frames to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Load Haarcascades for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return False   # No faces found in the frame

    # Extract the first face found in the frame
    x, y, w, h = faces[0]
    roi_frame = gray_frame[y:y + h, x:x + w]

    # Resize the reference image to match the size of the detected face
    resized_reference_image = cv2.resize(gray_reference_image, (w, h))

    # Compare the faces using structural similarity index
    ssim_index = ssim(roi_frame, resized_reference_image)

    return ssim_index > 0.7  # Adjust the threshold as needed
