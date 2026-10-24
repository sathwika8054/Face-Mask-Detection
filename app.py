
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import uuid

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mask_detector.h5"
PROTO_PATH = BASE_DIR / "face_detector" / "deploy.prototxt"
WEIGHTS_PATH = BASE_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"

RESULT_DIR = BASE_DIR / "static" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

model = load_model(str(MODEL_PATH))
face_detector = cv2.dnn.readNetFromCaffe(
    str(PROTO_PATH), str(WEIGHTS_PATH)
)


def detect_mask(image):
    output = image.copy()
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array(
            [width, height, width, height]
        )
        x1, y1, x2, y2 = box.astype("int")

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        scores = model.predict(face, verbose=0)[0]
        mask_score, no_mask_score = scores[:2]

        if mask_score > no_mask_score:
            label = "Mask"
            color = (0, 200, 0)
            score = mask_score
        else:
            label = "No Mask"
            color = (0, 0, 255)
            score = no_mask_score

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{label}: {score * 100:.1f}%",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    return output


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("image")

    if not uploaded or not uploaded.filename:
        return "Please select an image.", 400

    data = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        return "Invalid image file.", 400

    result = detect_mask(image)
    filename = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(RESULT_DIR / filename), result)

    return render_template(
        "index.html",
        result="Detection complete!",
        image_path=f"/static/results/{filename}"
    )


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True) or {}
    frame_data = data.get("image", "")

    try:
        encoded = frame_data.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded)
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify({"error": "Invalid camera frame"}), 400

        result = detect_mask(image)
        success, buffer = cv2.imencode(".jpg", result)

        if not success:
            return jsonify({"error": "Could not encode image"}), 500

        result_base64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"image": result_base64})

    except Exception as error:
        app.logger.exception("Frame detection failed")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True)
