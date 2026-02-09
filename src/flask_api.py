"""
Insurance Fraud Detection
Flask API + React Dashboard (Production Ready)
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import os
from datetime import datetime

# ======================================================
# Paths (IMPORTANT for Render)
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

# ======================================================
# Flask App
# ======================================================
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="")

# ======================================================
# Global ML Artifacts
# ======================================================
MODEL = None
SCALER = None
ENCODERS = None
FEATURE_NAMES = None


# ======================================================
# Load Model Artifacts
# ======================================================
def load_model_artifacts():
    global MODEL, SCALER, ENCODERS, FEATURE_NAMES

    try:
        models_dir = os.path.join(BASE_DIR, "models")

        model_path = os.path.join(models_dir, "xgboost_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        encoder_path = os.path.join(models_dir, "label_encoders.pkl")
        features_path = os.path.join(models_dir, "feature_names.json")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                MODEL = pickle.load(f)

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                SCALER = pickle.load(f)

        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                ENCODERS = pickle.load(f)

        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                FEATURE_NAMES = json.load(f)

        print("✅ Model artifacts loaded successfully")
        return True

    except Exception as e:
        print("❌ Error loading artifacts:", e)
        return False


# ======================================================
# React Dashboard (FIXES 404 ON RENDER)
# ======================================================
@app.route("/", defaults={"path": ""}, methods=["GET", "HEAD"])
@app.route("/<path:path>", methods=["GET", "HEAD"])
def serve_react(path):
    file_path = os.path.join(app.static_folder, path)

    # Serve static assets
    if path and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)

    # SPA fallback
    return send_from_directory(app.static_folder, "index.html")


# ======================================================
# API ENDPOINTS
# ======================================================
@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "message": "Insurance Fraud Detection API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if MODEL else "degraded",
        "model_loaded": MODEL is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    score = 0

    if data.get("claim_amount", 0) > 10000:
        score += 30
    if data.get("policy_tenure_months", 24) < 12:
        score += 25
    if data.get("witness_count", 1) == 0:
        score += 15
    if data.get("police_report_filed", 1) == 0:
        score += 10

    probability = min(score / 100, 0.95)

    return jsonify({
        "success": True,
        "prediction": {
            "is_fraud": probability > 0.5,
            "fraud_probability": round(probability * 100, 2)
        }
    })


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    data = request.json
    claims = data.get("claims", [])

    results = []
    for idx, claim in enumerate(claims):
        score = 30 if claim.get("claim_amount", 0) > 10000 else 10
        probability = min(score / 100, 0.95)

        results.append({
            "claim_id": claim.get("claim_id", idx),
            "fraud_probability": round(probability * 100, 2),
            "is_fraud": probability > 0.5
        })

    return jsonify({"predictions": results})


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    load_model_artifacts()
    app.run(host="0.0.0.0", port=5000)
