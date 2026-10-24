
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2
import os
import time


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            if endX <= startX or endY <= startY:
                continue

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if faces:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=1, verbose=0)
    else:
        preds = []

    return locs, preds


# Face detector files
prototxtPath = os.path.join(
    "face_detector", "deploy.prototxt"
)

weightsPath = os.path.join(
    "face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

if not os.path.exists(prototxtPath):
    raise FileNotFoundError(f"Missing file: {prototxtPath}")

if not os.path.exists(weightsPath):
    raise FileNotFoundError(f"Missing file: {weightsPath}")

# Load face detector
faceNet = cv2.dnn.readNetFromCaffe(
    prototxtPath, weightsPath
)

# Load mask detection model
maskNet = load_model(
    "mask_detector.h5", compile=False
)

# Use CPU for face detection
faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("[INFO] Starting webcam...")

# Start camera
vs = VideoStream(src=0).start()
time.sleep(2)

try:
    while True:
        frame = vs.read()

        if frame is None:
            print("[ERROR] Camera frame not received.")
            break

        # Keep the processing size small for smoother video
        frame = cv2.resize(frame, (640, 480))

        (locs, preds) = detect_and_predict_mask(
            frame, faceNet, maskNet
        )

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # Assumes model output index 0 = Mask,
            # index 1 = No Mask
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            confidence = max(mask, withoutMask) * 100

            text = "{}: {:.2f}%".format(
                label, confidence
            )

            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                color,
                2
            )

            cv2.putText(
                frame,
                text,
                (startX, max(startY - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Face Mask Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    vs.stop()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")
