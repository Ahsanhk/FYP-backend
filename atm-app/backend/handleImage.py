import cv2
import numpy as np


def process_image(username, image):
    if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return {"error": "Unsupported image format"}
    image_array = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), -1)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(
        image_array, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return {"error": "No faces found in the image"}

    (x, y, w, h) = faces[0]
    cropped_image = image_array[y:y+h, x:x+w]

    cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image_csv = cropped_image_gray.flatten()

    return {
        "username": username,
        "image_data": cropped_image_csv.tolist()
    }
