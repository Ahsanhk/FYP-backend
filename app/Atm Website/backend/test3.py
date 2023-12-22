from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from deepface import DeepFace
import time

app = FastAPI()

net = cv2.dnn.readNet("yolo/yolov7.weights", "yolo/yolov7.cfg")
layer_names = net.getUnconnectedOutLayersNames()

emotion_model = DeepFace.build_model('Emotion')


def detect_emotion(frame):
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

    # Analyze emotion only for the first detected face
    if len(boxes) > 0:
        x, y, w, h = boxes[0]
        face = frame[y:y+h, x:x+w]

        if face.size > 0:
            # Analyze all emotions
            emotion_results = DeepFace.analyze(
                face, actions=['emotion'], enforce_detection=False)

            if emotion_results and 'dominant_emotion' in emotion_results[0]:
                dominant_emotion = emotion_results[0]['dominant_emotion']
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (
                    x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if dominant_emotion.lower() == 'fear':
                    return True

    return False


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise HTTPException(
            status_code=500, detail="Error: Could not open camera.")

    try:
        start_time = time.time()
        while time.time() - start_time < 30:
            ret, frame = cap.read()

            if not ret or frame is None:
                raise HTTPException(
                    status_code=500, detail="Error: Could not read frame.")

            stop_detection = detect_emotion(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _, img_encoded = cv2.imencode('.png', frame_rgb)

            img_bytes = img_encoded.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')

            if stop_detection:
                break

    finally:

        cap.release()
