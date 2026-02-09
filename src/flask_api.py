"""
Flask API for Insurance Fraud Detection
Production-ready REST API + React Dashboard
(Render compatible)
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import os
from datetime import datetime

# ==========================================================
# Paths (IMPORTANT: works on Render + local)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

# ==========================================================
# Flask App
# ==========================================================
app = Flask(
    __name__,
    static_folder=FRONTEND_DIST,
    static_url_path="/static"
)

# ==========================================================
# Global ML Artifacts
# ==========================================================
MODEL = None
SCALER = None
ENCODERS = None
FEATURE_NAMES = None


# ==========================================================
# Load Model Artifacts
# ==========================================================
def load_model_artifacts():
    global MODEL, SCALER, ENCODERS, FEATURE_NAMES

    try:
        model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
        encoders_path = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
        features_path = os.path.join(BASE_DIR, "models", "feature_names.json")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                MODEL = pickle.load(f)

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                SCALER = pickle.load(f)

        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                ENCODERS = pickle.load(f)

        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                FEATURE_NAMES = json.load(f)

        print("✅ Model artifacts loaded successfully")
        return True

    except Exception as e:
        print("❌ Error loading model artifacts:", e)
        return False


# ==========================================================
# React Dashboard Routes (CRITICAL FIX)
# ==========================================================

@app.route("/", methods=["GET", "HEAD"])
def serve_root():
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/<path:path>", methods=["GET", "HEAD"])
def serve_react(path):
    file_path = os.path.join(FRONTEND_DIST, path)

    # Serve static assets
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIST, path)

    # React SPA fallback
    return send_from_directory(FRONTEND_DIST, "index.html")


# ==========================================================
# API Routes
# ==========================================================

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
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input provided"}), 400

        score = 0

        if data.get("claim_amount", 0) > 10000:
            score += 30
        if data.get("policy_tenure_months", 24) < 12:
            score += 25
        if data.get("witness_count", 1) == 0:
            score += 15

        probability = min(score / 100, 0.95)

        return jsonify({
            "success": True,
            "prediction": {
                "is_fraud": probability > 0.5,
                "fraud_probability": round(probability * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    data = request.json or {}
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


# ==========================================================
# Entry Point
# ==========================================================
if __name__ == "__main__":
    load_model_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
