import cv2
import os
import tensorflow as tf
import numpy as np
from IPython.display import display, Image
from PIL import Image as PILImage
from deepface import DeepFace

net = cv2.dnn.readNet("yolo/yolov7.weights", "yolo/yolov7.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getUnconnectedOutLayersNames()


def detect_faces():
    cascade_path = os.path.join(
        r'C:\Users\ahsan\Final Year Project\Atm Website\backend\assets\haar-cascade-files-master',
        'haarcascade_frontalface_default.xml'
    )

    consecutive_multiple_faces = 0
    consecutive_fearful_anxious = 0

    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    multiple_faces = False
    fearful_or_anxious_detected = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # frame_count += 1
        # # Process every 3rd frame
        # if frame_count % 3 == 0:
        height, width, _ = frame.shape

        # multiple face detection starts here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 1:

            # return True
            consecutive_multiple_faces += 1
            consecutive_fearful_anxious = 0
            if consecutive_multiple_faces >= 4:
                print("multiple faces detected!!!")
                multiple_faces = True
                break
            else:
                consecutive_multiple_faces = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # emotion recognition starts here
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        # Reset bounding boxes
        bounding_boxes = []

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
                    bounding_boxes.append((x, y, x + w, y + h))

        # Analyzeing emotions for each bounding box
        for (x, y, x2, y2) in bounding_boxes:
            face = frame[y:y2, x:x2]
            emotion_results = DeepFace.analyze(
                face, actions=['emotion'], enforce_detection=False)

            if emotion_results:
                dominant_emotion = emotion_results[0]['dominant_emotion']
                cv2.rectangle(
                    frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check emotion here
                if dominant_emotion.lower() in ['fear', 'anxious']:
                    consecutive_fearful_anxious += 1
                    # consecutive_multiple_faces = 0

                    if consecutive_fearful_anxious >= 4:
                        print("fearful emotion true!!!")
                        fearful_or_anxious_detected = True
                        break
                        return fearful_or_anxious_detected
                else:
                    consecutive_fearful_anxious = 0

        cv2.imshow('Live Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27 or multiple_faces or fearful_or_anxious_detected:
            break

    cap.release()
    cv2.destroyAllWindows()
    if multiple_faces:
        return multiple_faces
    elif fearful_or_anxious_detected:
        return fearful_or_anxious_detected
    print("multiple faces detected!!!")


if __name__ == "__main__":
    detect_faces()
