🛡️ Insurance Fraud Claim Detection using Predictive Analytics

🚀 Live Deployed Application
👉 https://insurance-fraud-detection-yskl.onrender.com

📊 Project Overview

This project is an end-to-end Machine Learning + Full Stack application that detects fraudulent insurance claims using predictive analytics.

It demonstrates the complete lifecycle of a real-world ML system:

Data preprocessing & feature engineering

Model training and evaluation

REST API development

Interactive dashboard

Cloud deployment

The system helps insurance companies identify high-risk claims early, reduce financial losses, and improve investigation efficiency.

🧭 End-to-End System Flow
Insurance Claims Dataset
        ↓
Data Cleaning & Validation
        ↓
Feature Engineering (Fraud Indicators)
        ↓
Model Training & Evaluation
        ↓
Best Model Selection (XGBoost)
        ↓
Saved Model Artifacts (.pkl, .json)
        ↓
Flask REST API
        ↓
React Dashboard (Visualization + Prediction)
        ↓
Cloud Deployment (Render)

🧠 Machine Learning Pipeline
1️⃣ Dataset Preparation

Handling missing values

Outlier treatment

Data type corrections

Encoding categorical variables

2️⃣ Feature Engineering

Key fraud-related features:

Claim-to-vehicle value ratio

Policy tenure risk flags

Evidence score

Claim reporting delay

Interaction-based fraud indicators

3️⃣ Model Training

Models evaluated:

Logistic Regression

Random Forest

Gradient Boosting

LightGBM

XGBoost (Selected Best Model)

4️⃣ Model Performance
Metric	Value
Accuracy	94%
ROC-AUC	97%
Precision	93%
Recall	92%
F1-Score	92%
🌐 Backend – Flask REST API

The backend exposes trained ML models through REST endpoints.

API Endpoints
Method	Endpoint	Description
GET	/	React Dashboard
GET	/api	API Information
GET	/health	Health Check
POST	/predict	Single Claim Prediction
POST	/batch-predict	Batch Claim Predictions
Sample Prediction Input
{
  "age": 35,
  "claim_amount": 15000,
  "policy_tenure_months": 6,
  "witness_count": 0
}

🎨 Frontend – React Dashboard

The frontend provides a user-friendly visualization layer.

Dashboard Features

Fraud risk KPI cards

Claim fraud prediction form

Bar, Pie & Line charts

Real-time API integration

Responsive and professional UI

Frontend Stack

React (Vite)

Tailwind CSS

REST API integration

☁️ Deployment Details

Platform: Render

Backend Server: Gunicorn + Flask

Frontend: React production build served via Flask

🔗 Live Application URL:
https://insurance-fraud-detection-yskl.onrender.com

📁 Project Structure
insurance_fraud_detection/
│
├── frontend/
│   └── dist/                  # React production build
│
├── src/
│   ├── fraud_detection_complete.py   # ML training pipeline
│   └── flask_api.py                  # Flask API + React serving
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
│
├── outputs/                   # Charts, EDA & reports
├── data/                      # Dataset
├── requirements.txt
└── README.md

📘 Project Documentation

📄 A detailed 10–15 page PDF documentation is included separately, covering:

Architecture diagrams

Data flow & ML pipeline

Model evaluation

API workflow

Deployment steps

💼 Business Impact

Early detection of fraudulent claims

Reduced financial losses

Faster claim investigation

Improved operational efficiency

💰 Estimated Net Business Benefit: $13.5 Million

🛠️ Technologies Used
Machine Learning

Python

Scikit-learn

XGBoost

Pandas, NumPy

Backend

Flask

Gunicorn

Frontend

React

Tailwind CSS

Deployment & Tools

Render

GitHub

👤 Author

Goutham Reddy Gopi Reddy
Machine Learning & Full Stack Developer

📄 License

This project is licensed under the MIT License.
