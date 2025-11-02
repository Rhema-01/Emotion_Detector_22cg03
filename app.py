from flask import Flask, request, jsonify, render_template, make_response
import os, base64, io, traceback
from PIL import Image
import joblib
import numpy as np

# optional CORS
try:
    from flask_cors import CORS
except Exception:
    CORS = None

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB

if CORS:
    CORS(app)

# model paths
MODEL_PATH = os.path.join("models", "mlp_emotion.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
LE_PATH = os.path.join("models", "label_encoder.pkl")

mlp_model = None
scaler = None
label_encoder = None

# load model/scaler if present
try:
    if os.path.exists(MODEL_PATH):
        mlp_model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(LE_PATH):
        label_encoder = joblib.load(LE_PATH)

    print("Model loaded:", bool(mlp_model))
    print("Scaler loaded:", bool(scaler))
    print("Label Encoder loaded:", bool(label_encoder))
    if label_encoder is not None:
        print("Model classes:", list(label_encoder.classes_))
except Exception as e:
    print("Error loading model/scaler:", e)
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_status', methods=['GET'])
def model_status():
    return jsonify({
        "model_loaded": mlp_model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": label_encoder is not None,
        "classes": list(label_encoder.classes_) if label_encoder is not None else []
    })

def preprocess_image_bytes(img_bytes, img_size=(48,48)):
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
    img = img.resize(img_size)
    arr = np.asarray(img, dtype=np.float32).flatten()
    return arr

@app.route('/detect_emotion', methods=['POST', 'OPTIONS'])
def detect_emotion():
    # handle CORS preflight
    if request.method == 'OPTIONS':
        resp = make_response(('', 200))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    if not data or 'image' not in data:
        return jsonify({"error": 'Missing "image" field'}), 400

    try:
        if mlp_model is None or scaler is None or label_encoder is None:
            return jsonify({"error": "Model, scaler, or encoder not loaded on the server.",
                            "model_loaded": mlp_model is not None,
                            "scaler_loaded": scaler is not None,
                            "encoder_loaded": label_encoder is not None}), 503 # Service Unavailable

        b64 = data['image']
        if ',' in b64:
            b64 = b64.split(',',1)[1]
        img_bytes = base64.b64decode(b64)
        arr = preprocess_image_bytes(img_bytes, img_size=(48,48))
        X = scaler.transform([arr])
        probs = mlp_model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        
        # Convert NumPy types to standard Python types for JSON serialization
        label = str(mlp_model.classes_[idx])
        confidence = float(probs[idx]) 
        resp = jsonify({"emotion": label, "confidence": confidence})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        tb = traceback.format_exc()
        print("Prediction error:", tb)
        return jsonify({"error": "Server error processing image", "details": str(e),
                        "trace_tail": tb.splitlines()[-6:]}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)