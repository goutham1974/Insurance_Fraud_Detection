
# 🛡️ Insurance Claim Fraud Detection using Predictive Analytics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-black)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB)](https://react.dev/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](.)

---

## 📌 Project Overview

End-to-end **Machine Learning + Full Stack** application to detect **fraudulent insurance claims**.

This project includes:

- 🧠 ML model training pipeline (XGBoost / LightGBM)
- 🌐 Flask REST API for predictions
- 🎨 React + Tailwind Dashboard UI
- 📊 KPI charts and fraud prediction form
- ☁️ Cloud-ready deployment (Render / AWS)

---

## 🧭 System Architecture

```
Dataset (CSV)
     ↓
Data Cleaning + Feature Engineering
     ↓
Model Training (XGBoost)
     ↓
Saved Artifacts (.pkl, .json)
     ↓
Flask API (serves model)
     ↓
React Dashboard (served by Flask)
```

---

## 🚀 Quick Start (Full Stack)

### 1️⃣ Install Python dependencies

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2️⃣ Train model (creates models/)

```bash
python src/fraud_detection_complete.py
```

### 3️⃣ Build React dashboard (VERY IMPORTANT)

```bash
cd frontend
npm install
npm run build
```

This creates:

```
frontend/dist/index.html
```

### 4️⃣ Start the Flask server

```bash
cd ..
python src/flask_api.py
```

### 5️⃣ Open in browser

👉 http://localhost:5000/

You will see the **Fraud Detection Dashboard UI**.

---

## 📁 Project Structure

```
insurance_fraud_detection/
│
├── frontend/                 # React Dashboard (Vite + Tailwind)
│   └── dist/                 # Production build served by Flask
│
├── src/
│   ├── fraud_detection_complete.py  # ML pipeline
│   └── flask_api.py                 # Flask + React server
│
├── models/                   # Saved ML artifacts
├── outputs/                  # EDA, metrics, charts
├── data/                     # Dataset
├── requirements.txt
└── README.md
```

---

## 🤖 ML Model Pipeline

| Step | Description |
|-----|-------------|
| Data Cleaning | Handle missing values, fix types |
| Feature Engineering | 40+ derived fraud features |
| Encoding | LabelEncoder for categorical data |
| Scaling | RobustScaler |
| Balancing | SMOTETomek |
| Models | Logistic, RF, XGBoost, LightGBM |
| Best Model | XGBoost (94% accuracy) |
| Saved Artifacts | model.pkl, scaler.pkl, encoders.pkl |

---

## 🌐 Flask API Endpoints

| Method | Endpoint | Purpose |
|-------|----------|---------|
| GET | `/` | React Dashboard |
| GET | `/api` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/batch-predict` | Batch prediction |

---

## 🎨 Dashboard Features

- KPI Cards (Fraud %, Total Claims, Risk Levels)
- Fraud Prediction Form
- Charts (Bar, Pie, Line)
- Professional UI
- Real-time API integration

---

## 🧪 Example API Call

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"age":35,"claim_amount":15000,"policy_tenure_months":6}'
```

---

## ☁️ Deployment on Render (Recommended)

### Step 1: Push to GitHub

### Step 2: On Render → New Web Service

| Setting | Value |
|---------|------|
| Environment | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn src.flask_api:app` |

### Step 3: Done 🎉

Your app will be live like:

```
https://insurance-fraud.onrender.com
```

---

## 💰 Business Impact

| Metric | Value |
|-------|------|
| Accuracy | 94% |
| ROC-AUC | 97% |
| Fraud Detection | 92% |
| Net Benefit | $13.5M |

---

## 🛠 Technologies Used

**Backend:** Python, Flask, Scikit-learn, XGBoost  
**Frontend:** React, Vite, Tailwind CSS  
**ML:** Feature Engineering, SMOTE, Cross Validation  
**Deployment:** Gunicorn, Render  

---

## 🐛 Troubleshooting

**Model not found?**
```
python src/fraud_detection_complete.py
```

**Dashboard not loading?**
```
cd frontend
npm run build
```

---

## 📄 License

MIT License

---

## 🙌 Author

Goutham Reddy Gopi Reddy
