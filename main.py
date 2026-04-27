"""
app.py
──────
Flask REST API for the Credit Card Compliance Checker.
Endpoints:
  POST /predict  – returns prediction + confidence + feature importance
  GET  /metadata – returns model metadata / leaderboard
  GET  /health   – liveness check
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # allow frontend on any origin during development

# ── Load artefacts at startup ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "models"
model = joblib.load(MODEL_DIR / "best_model.pkl")
if hasattr(model, "named_steps") and "clf" in model.named_steps:
    model.named_steps["clf"].set_params(n_jobs=1)

with open(MODEL_DIR / "metadata.json") as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]
COMPLIANCE_RULES = metadata["compliance_rules"]


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/index.html", methods=["GET"])
def index_html():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── Rule-level insight generator ──────────────────────────────────────────────
def rule_check(data: dict) -> list[dict]:
    """
    Returns a list of {field, status, message} dicts for each feature.
    status: 'pass' | 'fail'
    """
    insights = []

    checks = [
        (
            "interest_rate",
            lambda v: 12 <= v <= 42,
            f"Interest rate {data['interest_rate']}% is within allowed 12%–42% range.",
            f"Interest rate {data['interest_rate']}% is OUTSIDE the allowed 12%–42% range.",
        ),
        (
            "late_payment_fee",
            lambda v: v <= 1200,
            f"Late payment fee ₹{data['late_payment_fee']} is within ₹1,200 limit.",
            f"Late payment fee ₹{data['late_payment_fee']} EXCEEDS the ₹1,200 limit.",
        ),
        (
            "annual_fee",
            lambda v: v <= 5000,
            f"Annual fee ₹{data['annual_fee']} is within ₹5,000 limit.",
            f"Annual fee ₹{data['annual_fee']} EXCEEDS the ₹5,000 limit.",
        ),
        (
            "billing_cycle",
            lambda v: 25 <= v <= 45,
            f"Billing cycle {data['billing_cycle']} days is within 25–45 day window.",
            f"Billing cycle {data['billing_cycle']} days is OUTSIDE the 25–45 day window.",
        ),
        (
            "minimum_payment_pct",
            lambda v: 2 <= v <= 10,
            f"Minimum payment {data['minimum_payment_pct']}% is within 2%–10% range.",
            f"Minimum payment {data['minimum_payment_pct']}% is OUTSIDE the 2%–10% range.",
        ),
        (
            "disclosure_provided",
            lambda v: v == 1,
            "Required disclosure has been provided.",
            "Required disclosure is MISSING — mandatory for compliance.",
        ),
    ]

    for field, rule, ok_msg, fail_msg in checks:
        val = data[field]
        passed = rule(val)
        insights.append({
            "field":   field,
            "value":   val,
            "status":  "pass" if passed else "fail",
            "message": ok_msg if passed else fail_msg,
        })

    return insights


# ── /predict ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)

        # Extract & validate inputs
        data = {
            "interest_rate":       float(body["interest_rate"]),
            "late_payment_fee":    float(body["late_payment_fee"]),
            "annual_fee":          float(body["annual_fee"]),
            "billing_cycle":       float(body["billing_cycle"]),
            "minimum_payment_pct": float(body["minimum_payment_pct"]),
            "disclosure_provided": int(body["disclosure_provided"]),
        }

        X = pd.DataFrame([[data[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

        # Model prediction
        pred_label = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0]
        confidence = round(float(max(proba)) * 100, 2)

        prediction_text = "Compliant" if pred_label == 1 else "Non-Compliant"

        # Feature importances (normalised)
        feat_imp = metadata["feature_importance"]

        # Rule-level insights
        rule_insights = rule_check(data)
        violations = [r for r in rule_insights if r["status"] == "fail"]

        response = {
            "prediction":         prediction_text,
            "prediction_label":   pred_label,
            "confidence":         confidence,
            "probabilities": {
                "non_compliant": round(float(proba[0]) * 100, 2),
                "compliant":     round(float(proba[1]) * 100, 2),
            },
            "feature_importance": feat_imp,
            "rule_insights":      rule_insights,
            "violations":         violations,
            "total_violations":   len(violations),
            "model_used":         metadata["best_model"],
        }

        return jsonify(response), 200

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /metadata ─────────────────────────────────────────────────────────────────
@app.route("/metadata", methods=["GET"])
def get_metadata():
    return jsonify(metadata), 200


# ── /health ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": metadata["best_model"]}), 200


if __name__ == "__main__":
    print(f"🚀  Starting Compliance API  |  Model: {metadata['best_model']}")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
