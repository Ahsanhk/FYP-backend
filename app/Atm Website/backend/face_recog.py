from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import requests
from PIL import Image
from io import BytesIO
import cv2
import face_recognition
import numpy as np
import time
import matplotlib.pyplot as plt

app = FastAPI()


def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


async def face_detection():
    reference_image_url = "https://as2.ftcdn.net/v2/jpg/01/29/38/45/1000_F_129384512_davV17UpN59sMn3u3SBhnZIPSyUKJ0Cn.jpg"
    reference_image = load_image_from_url(reference_image_url)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]

    cap = cv2.VideoCapture(0)

    duration = 2
    start_time = time.time()

    while (time.time() - start_time) < duration:

        ret, frame = cap.read()
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
                    return "Match found!"

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    cap.release()
    cv2.destroyAllWindows()
    print("No match found within {} seconds.".format(duration))
    return "No match found within {} seconds.".format(duration)


@app.get("/", response_class=HTMLResponse)
async def face_detection_route(request: Request):
    result = await face_detection()
    return f"<html><body><h1>{result}</h1></body></html>"
