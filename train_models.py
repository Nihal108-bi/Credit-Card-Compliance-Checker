"""
train_models.py
───────────────
Trains three classifiers on the synthetic compliance dataset:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest

Evaluates each on Accuracy, F1, ROC-AUC → saves the best model + metadata.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

# ── Local generator ──────────────────────────────────────────────────────────
from generate_data import generate_dataset

FEATURE_COLS = [
    "interest_rate",
    "late_payment_fee",
    "annual_fee",
    "billing_cycle",
    "minimum_payment_pct",
    "disclosure_provided",
]
TARGET_COL = "label"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Build dataset ─────────────────────────────────────────────────────────────
def load_data():
    csv_path = Path("dataset.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("Loaded existing dataset.csv")
    else:
        df = generate_dataset(300)
        df.to_csv(csv_path, index=False)
        print("Generated fresh dataset.csv")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── Model definitions ─────────────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=7,
                random_state=42, n_jobs=-1
            )),
        ]),
    }


# ── Evaluate a single model ───────────────────────────────────────────────────
def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_proba)
    cv_f1 = cross_val_score(model, X_train, y_train,
                             cv=5, scoring="f1").mean()

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  CV F1 (5) : {cv_f1:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Compliant", "Compliant"]))

    return {
        "name":        name,
        "accuracy":    round(acc,   4),
        "f1":          round(f1,    4),
        "roc_auc":     round(auc,   4),
        "cv_f1_mean":  round(cv_f1, 4),
        "composite":   round((acc + f1 + auc) / 3, 4),   # ranking metric
    }


# ── Feature importance helper ─────────────────────────────────────────────────
def get_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):           # Tree / Forest
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):                         # Logistic Regression
        imp = np.abs(clf.coef_[0])
    else:
        imp = np.ones(len(feature_names)) / len(feature_names)

    imp_norm = imp / imp.sum()
    return {f: round(float(v), 4) for f, v in zip(feature_names, imp_norm)}


# ── Main training pipeline ────────────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test = load_data()
    models  = build_models()
    results = []

    for name, model in models.items():
        res = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append((res, model))

    # Pick best by composite score
    results.sort(key=lambda x: x[0]["composite"], reverse=True)
    best_meta, best_model = results[0]
    best_name = best_meta["name"]

    print(f"\n{'*'*50}")
    print(f"  ✅  Best Model : {best_name}")
    print(f"       Composite : {best_meta['composite']}")
    print(f"{'*'*50}\n")

    # Save best model
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    # Save feature importance
    feat_imp = get_feature_importance(best_model, FEATURE_COLS)

    # Build full leaderboard
    leaderboard = [
        {
            "rank":     i + 1,
            "name":     r["name"],
            "accuracy": r["accuracy"],
            "f1":       r["f1"],
            "roc_auc":  r["roc_auc"],
            "cv_f1":    r["cv_f1_mean"],
            "composite":r["composite"],
            "is_best":  r["name"] == best_name,
        }
        for i, (r, _) in enumerate(results)
    ]

    metadata = {
        "best_model":          best_name,
        "best_metrics":        best_meta,
        "feature_names":       FEATURE_COLS,
        "feature_importance":  feat_imp,
        "leaderboard":         leaderboard,
        "compliance_rules": {
            "interest_rate":       "Must be between 12% and 42%",
            "late_payment_fee":    "Must be ≤ ₹1,200",
            "annual_fee":          "Must be ≤ ₹5,000",
            "billing_cycle":       "Must be 25–45 days",
            "minimum_payment_pct": "Must be between 2% and 10%",
            "disclosure_provided": "Must be Yes (1)",
        },
    }

    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: models/best_model.pkl")
    print("Saved: models/metadata.json")
    print(f"\nFeature Importance ({best_name}):")
    for feat, score in sorted(feat_imp.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        print(f"  {feat:<25} {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
