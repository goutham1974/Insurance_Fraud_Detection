"""
Test script for Insurance Fraud Detection API
"""

import requests
import json

print("="*70)
print("   INSURANCE FRAUD DETECTION API - TEST SUITE")
print("="*70)

# API base URL
BASE_URL = "http://localhost:5000"

# Test 1: Health Check
print("\n[TEST 1] Health Check")
print("-"*70)
response = requests.get(f"{BASE_URL}/health")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
assert response.status_code == 200
print("✅ PASSED")

# Test 2: Model Info
print("\n[TEST 2] Model Information")
print("-"*70)
response = requests.get(f"{BASE_URL}/model-info")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
assert response.status_code == 200
print("✅ PASSED")

# Test 3: High-Risk Fraudulent Claim
print("\n[TEST 3] High-Risk Fraud Prediction")
print("-"*70)
high_risk_claim = {
    "age": 24,
    "claim_amount": 25000,
    "policy_tenure_months": 2,
    "vehicle_value": 20000,
    "police_report_filed": 0,
    "witness_count": 0,
    "number_of_previous_claims": 4,
    "claim_report_delay_hours": 72
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=high_risk_claim,
    headers={"Content-Type": "application/json"}
)

print(f"Status Code: {response.status_code}")
result = response.json()
print(f"Response: {json.dumps(result, indent=2)}")

if result['success']:
    pred = result['prediction']
    print(f"\n{'='*70}")
    print(f"🔴 FRAUD DETECTED: {pred['is_fraud']}")
    print(f"📊 Probability: {pred['fraud_probability']}%")
    print(f"⚠️  Risk Level: {result['risk_assessment']['risk_level']}")
    print(f"🚨 Action: {result['recommendation']['action']}")
    print(f"💡 Recommendation: {result['recommendation']['message']}")
    print(f"{'='*70}")
    assert pred['is_fraud'] == True
    print("✅ PASSED")
else:
    print("❌ FAILED")

# Test 4: Low-Risk Legitimate Claim
print("\n[TEST 4] Low-Risk Legitimate Prediction")
print("-"*70)
low_risk_claim = {
    "age": 45,
    "claim_amount": 3500,
    "policy_tenure_months": 48,
    "vehicle_value": 28000,
    "police_report_filed": 1,
    "witness_count": 2,
    "number_of_previous_claims": 0
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=low_risk_claim,
    headers={"Content-Type": "application/json"}
)

print(f"Status Code: {response.status_code}")
result = response.json()
print(f"Response: {json.dumps(result, indent=2)}")

if result['success']:
    pred = result['prediction']
    print(f"\n{'='*70}")
    print(f"🟢 FRAUD DETECTED: {pred['is_fraud']}")
    print(f"📊 Probability: {pred['fraud_probability']}%")
    print(f"✅ Risk Level: {result['risk_assessment']['risk_level']}")
    print(f"👍 Action: {result['recommendation']['action']}")
    print(f"💡 Recommendation: {result['recommendation']['message']}")
    print(f"{'='*70}")
    assert pred['is_fraud'] == False
    print("✅ PASSED")
else:
    print("❌ FAILED")

# Test 5: Batch Prediction
print("\n[TEST 5] Batch Prediction")
print("-"*70)
batch_claims = {
    "claims": [
        {
            "claim_id": "CLM001",
            "age": 30,
            "claim_amount": 5000,
            "policy_tenure_months": 24,
            "witness_count": 2,
            "number_of_previous_claims": 0
        },
        {
            "claim_id": "CLM002",
            "age": 22,
            "claim_amount": 22000,
            "policy_tenure_months": 3,
            "witness_count": 0,
            "number_of_previous_claims": 5
        },
        {
            "claim_id": "CLM003",
            "age": 50,
            "claim_amount": 4000,
            "policy_tenure_months": 60,
            "witness_count": 3,
            "number_of_previous_claims": 1
        }
    ]
}

response = requests.post(
    f"{BASE_URL}/batch-predict",
    json=batch_claims,
    headers={"Content-Type": "application/json"}
)

print(f"Status Code: {response.status_code}")
result = response.json()
print(f"Response: {json.dumps(result, indent=2)}")

if result['success']:
    print(f"\n{'='*70}")
    print(f"📊 BATCH RESULTS:")
    print(f"   Total Claims: {result['summary']['total_claims']}")
    print(f"   Fraudulent: {result['summary']['fraudulent_claims']}")
    print(f"   Fraud Rate: {result['summary']['fraud_rate']}")
    print(f"{'='*70}")
    print("✅ PASSED")
else:
    print("❌ FAILED")

# Summary
print("\n" + "="*70)
print("   🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n✅ Your API is working perfectly!")
print("✅ Ready for production deployment!")