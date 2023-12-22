import cv2
import numpy as np
from IPython.display import display, Image
from PIL import Image as PILImage
from deepface import DeepFace

net = cv2.dnn.readNet("yolo/yolov7.weights", "yolo/yolov7.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getUnconnectedOutLayersNames()


def detect_emotion():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        try:
            fearful_or_anxious_detected = False
            frame_count = 0

            while not fearful_or_anxious_detected:
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Error: Could not read frame.")
                    break

                frame_count += 1

                if frame_count % 3 == 0:
                    frame = cv2.flip(frame, 1)
                    height, width, _ = frame.shape

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

                            if dominant_emotion.lower() in ['fear', 'anxious']:
                                fearful_or_anxious_detected = True
                                break

                    display(PILImage.fromarray(frame))
                    cv2.imshow('Live Emotion Detection', frame)

                if cv2.waitKey(1) & 0xFF == 27 or fearful_or_anxious_detected:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    return fearful_or_anxious_detected


# Call the optimized function
# result = detect_emotion()
