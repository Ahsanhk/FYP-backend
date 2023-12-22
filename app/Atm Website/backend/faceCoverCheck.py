import cv2
import numpy as np


def isFaceCovered():
    net = cv2.dnn.readNetFromDarknet("yolo/yolov7.cfg", "yolo/yolov7.weights")
    layer_names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(0)

    face_covered = False

    for i in range(10):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True)
        net.setInput(blob)
        outs = net.forward(layer_names)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.2 and class_id == 0:

                    if face_covered_detection(frame, detection):
                        face_covered = True

        cv2.imshow('Face Cover Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if face_covered:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return True if the face is covered, False otherwise
    return face_covered


def face_covered_detection(frame, detection):
    frame_height, frame_width, _ = frame.shape

    x_min = int(round(detection[0] * frame_width))
    y_min = int(round(detection[1] * frame_height))
    x_max = int(round(detection[2] * frame_width))
    y_max = int(round(detection[3] * frame_height))

    # print('x_min: ', x_min)
    # print('x_max: ', x_max)
    # print('y_min: ', y_min)
    # print('y_max: ', y_max)
    # print('break')

    if x_max > x_min and y_max > y_min:
        face_crop = frame[y_min:y_max, x_min:x_max]
        print('face_crop:', face_crop)

        if not face_crop.size == 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(
                gray, 127, 255, cv2.THRESH_BINARY_INV)

            white_pixels = np.sum(binary_image == 255)
            print('white_pixels:', white_pixels)

            if white_pixels > 0.7 * binary_image.size:
                return True

    return False
