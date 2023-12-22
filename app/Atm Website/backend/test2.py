# import cv2
# import numpy as np
# from typing import Optional
# from fastapi import FastAPI, Request
# from models.experimental import attempt_load

# app = FastAPI()


# def face_detection_multiple(frame: np.ndarray) -> Optional[np.ndarray]:
#     # Load the YOLOv7 model
#     net = attempt_load('yolov7.pt')
#     layer_names = net.getUnconnectedOutLayersNames()

#     # Convert the frame to a blob for YOLOv7 input
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True)
#     net.setInput(blob)

#     # Run YOLOv7 inference
#     outs = net.forward(layer_names)

#     # Process the detection results
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.2 and class_id == 0:  # Check if it's a face detection
#                 # Extract the bounding box coordinates of the detected face
#                 x_min = int(round(detection[0] * frame.shape[1]))
#                 y_min = int(round(detection[1] * frame.shape[0]))
#                 x_max = int(round(detection[2] * frame.shape[1]))
#                 y_max = int(round(detection[3] * frame.shape[0]))

#                 # Create a bounding box around the detected face
#                 cv2.rectangle(frame, (x_min, y_min),
#                               (x_max, y_max), (0, 255, 0), 2)

#     return frame
