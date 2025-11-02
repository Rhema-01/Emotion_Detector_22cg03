from __future__ import annotations
import base64
import io
import os
import traceback
from typing import Dict, Optional

from flask import Flask, request, jsonify, render_template, make_response
from PIL import Image
import joblib
import numpy as np

# Try CORS if available (optional)
try:
    from flask_cors import CORS  # type: ignore
except Exception:
    CORS = None  # type: ignore

# Flask app (templates/static explicit like original)
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

if CORS:
    CORS(app)

# Model file locations (same globals as your original)
MODEL_PATH = os.path.join("models", "mlp_emotion.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
LE_PATH = os.path.join("models", "label_encoder.pkl")

# these globals are intentionally kept with the same names
mlp_model: Optional[object] = None
scaler: Optional[object] = None
label_encoder: Optional[object] = None


def safe_load(path: str):
    """Load a joblib file if it exists, otherwise return None."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def initialize_model_environment() -> Dict[str, bool]:
    """Load model, scaler and encoder into module globals and return load status."""
    global mlp_model, scaler, label_encoder
    try:
        mlp_model = safe_load(MODEL_PATH)
        scaler = safe_load(SCALER_PATH)
        label_encoder = safe_load(LE_PATH)

        status = {
            "model_loaded": mlp_model is not None,
            "scaler_loaded": scaler is not None,
            "encoder_loaded": label_encoder is not None,
        }

        # print helpful info for the server logs
        print("Model loaded:", status["model_loaded"])
        print("Scaler loaded:", status["scaler_loaded"])
        print("Label Encoder loaded:", status["encoder_loaded"])
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            try:
                print("Model classes:", list(label_encoder.classes_))
            except Exception:
                pass
        return status
    except Exception as exc:
        print("Error loading model/scaler:", exc)
        traceback.print_exc()
        return {"model_loaded": False, "scaler_loaded": False, "encoder_loaded": False}


# initialize at import/run
initialize_model_environment()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/model_status", methods=["GET"])
def model_status():
    # safely build classes list only if encoder is present
    classes_list = []
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        try:
            classes_list = list(label_encoder.classes_)
        except Exception:
            classes_list = []

    return jsonify({
        "model_loaded": mlp_model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": label_encoder is not None,
        "classes": classes_list
    })


def bytes_to_grayscale_array(img_bytes: bytes, size: tuple[int, int] = (48, 48)) -> np.ndarray:
    """Convert image bytes -> flattened grayscale numpy array (float32)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32).flatten()
    return arr


def cors_preflight_response() -> "flask.wrappers.Response":
    """Return a minimal CORS preflight response (used in detect_emotion)."""
    resp = make_response(("", 200))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/detect_emotion", methods=["POST", "OPTIONS"])
def detect_emotion():
    # handle CORS
