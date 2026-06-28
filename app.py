import os
import sys
import types
import tempfile
import concurrent.futures
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# DYNAMIC MODULE MOCK FOR inference_sdk (Supports Python 3.14+)
# -----------------------------------------------------------------------------
mock_module = types.ModuleType("inference_sdk")

class InferenceHTTPClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def infer(self, image_path, model_id):
        import base64
        import requests
        
        with open(image_path, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode("utf-8")
            
        url = f"{self.api_url}/{model_id}?api_key={self.api_key}"
        response = requests.post(
            url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=base64_str
        )
        
        if response.status_code != 200:
            raise Exception(f"Roboflow API error {response.status_code}: {response.text}")
            
        return response.json()

mock_module.InferenceHTTPClient = InferenceHTTPClient
sys.modules["inference_sdk"] = mock_module

# -----------------------------------------------------------------------------
# FLASK BACKEND CONFIG & SETUP
# -----------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

from inference_sdk import InferenceHTTPClient

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not ROBOFLOW_API_KEY:
    print("[Backend] WARNING: ROBOFLOW_API_KEY not set – inference will be disabled.")
    CLIENT = None
else:
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )

THRESHOLD = 0.5

# Helper function to run a single inference
def run_model_inference(model_id, image_path):
    if CLIENT is None:
        # Inference disabled – return empty predictions
        return {"predictions": []}
    try:
        return CLIENT.infer(image_path, model_id=model_id)
    except Exception as e:
        print(f"[Backend] Error calling model {model_id}: {str(e)}")
        return {"predictions": []}

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def analyze_image(image_path):
    # 1. Run BOTH models in parallel to optimize latency
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_pothole = executor.submit(run_model_inference, "pothole-detection-i00zy/2", image_path)
        future_garbage = executor.submit(run_model_inference, "garbage-can-overflow/1", image_path)
        
        pothole_result = future_pothole.result()
        garbage_result = future_garbage.result()

    # 2. Extract highest confidence from each model (confidence = 0 if no predictions)
    pothole_predictions = pothole_result.get("predictions", [])
    pothole_confidence = max([p.get("confidence", 0) for p in pothole_predictions]) if pothole_predictions else 0.0

    garbage_predictions = garbage_result.get("predictions", [])
    garbage_confidence = max([g.get("confidence", 0) for g in garbage_predictions]) if garbage_predictions else 0.0

    print(f"[Backend] Pothole max confidence: {pothole_confidence}")
    print(f"[Backend] Garbage max confidence: {garbage_confidence}")

    max_confidence = max(pothole_confidence, garbage_confidence)

    # 3. Apply safety threshold first
    if pothole_confidence < THRESHOLD and garbage_confidence < THRESHOLD:
        return {
            "label": "unknown",
            "confidence": max_confidence,
            "message": "Low confidence detection"
        }

    # 4. Compare results to select highest confidence label
    if pothole_confidence > garbage_confidence:
        return {
            "label": "pothole",
            "confidence": pothole_confidence,
            "model": "pothole-detection-i00zy/2",
            "raw": pothole_result
        }
    else:
        return {
            "label": "garbage_overflow",
            "confidence": garbage_confidence,
            "model": "garbage-can-overflow/1",
            "raw": garbage_result
        }

# -----------------------------------------------------------------------------
# ENDPOINT
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "roboflow_ready": CLIENT is not None}), 200

@app.route("/analyze-image", methods=["POST"])
def analyze_image_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if CLIENT is None:
        return jsonify({"error": "ROBOFLOW_API_KEY not configured on server"}), 503
        
    # Save the file to a temporary location with a unique name
    import uuid
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)
    
    try:
        result = analyze_image(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Return expected structured response
        print(f"[Backend] Returning result: label={result['label']}, confidence={result['confidence']}")
        return jsonify({
            "label": result["label"],
            "confidence": float(result["confidence"])
        })
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[Backend] Error during analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
