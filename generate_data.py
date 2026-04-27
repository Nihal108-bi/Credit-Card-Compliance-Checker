"""
generate_data.py
────────────────
Synthetic Credit Card Compliance Dataset Generator
Uses Faker for realistic-looking data; labels are produced by
deterministic business rules so the ML models can learn patterns.
"""

import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_IN")   # India locale for ₹ relevance
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# COMPLIANCE RULE ENGINE  (ground truth labels)
# ──────────────────────────────────────────────
def is_compliant(row: dict) -> int:
    """
    Returns 1 (Compliant) or 0 (Non-Compliant).
    Rules mirror RBI / consumer-finance guidelines.
    """
    reasons = []

    # Rule 1 – Interest rate must be between 12% and 42%
    if not (12 <= row["interest_rate"] <= 42):
        reasons.append("interest_rate_out_of_range")

    # Rule 2 – Late payment fee ≤ ₹1 200
    if row["late_payment_fee"] > 1200:
        reasons.append("late_fee_too_high")

    # Rule 3 – Annual fee ≤ ₹5 000
    if row["annual_fee"] > 5000:
        reasons.append("annual_fee_too_high")

    # Rule 4 – Billing cycle 25–45 days
    if not (25 <= row["billing_cycle"] <= 45):
        reasons.append("billing_cycle_out_of_range")

    # Rule 5 – Minimum payment 2%–10%
    if not (2 <= row["minimum_payment_pct"] <= 10):
        reasons.append("min_payment_out_of_range")

    # Rule 6 – Disclosure must be provided
    if row["disclosure_provided"] == 0:
        reasons.append("no_disclosure")

    return 0 if reasons else 1


# ──────────────────────────────────────────────
# ROW GENERATOR
# ──────────────────────────────────────────────
def generate_row(force_compliant: bool | None = None) -> dict:
    """
    Generate a single synthetic record.
    force_compliant=True  → guaranteed compliant
    force_compliant=False → guaranteed non-compliant (random rule broken)
    force_compliant=None  → fully random
    """
    if force_compliant is True:
        row = {
            "interest_rate":       round(random.uniform(12, 42), 2),
            "late_payment_fee":    random.randint(100, 1200),
            "annual_fee":          random.randint(0, 5000),
            "billing_cycle":       random.randint(25, 45),
            "minimum_payment_pct": round(random.uniform(2, 10), 2),
            "disclosure_provided": 1,
        }
    elif force_compliant is False:
        row = {
            "interest_rate":       round(random.uniform(12, 42), 2),
            "late_payment_fee":    random.randint(100, 1200),
            "annual_fee":          random.randint(0, 5000),
            "billing_cycle":       random.randint(25, 45),
            "minimum_payment_pct": round(random.uniform(2, 10), 2),
            "disclosure_provided": 1,
        }
        # Break one or more rules
        num_violations = random.randint(1, 3)
        for _ in range(num_violations):
            violation = random.choice(["rate", "fee", "annual", "cycle", "minpay", "disc"])
            if violation == "rate":
                row["interest_rate"] = round(
                    random.choice([random.uniform(0, 11.9), random.uniform(42.1, 60)]), 2
                )
            elif violation == "fee":
                row["late_payment_fee"] = random.randint(1201, 2500)
            elif violation == "annual":
                row["annual_fee"] = random.randint(5001, 12000)
            elif violation == "cycle":
                row["billing_cycle"] = random.choice(
                    [random.randint(1, 24), random.randint(46, 90)]
                )
            elif violation == "minpay":
                row["minimum_payment_pct"] = round(
                    random.choice([random.uniform(0, 1.9), random.uniform(10.1, 20)]), 2
                )
            else:
                row["disclosure_provided"] = 0
    else:
        row = {
            "interest_rate":       round(random.uniform(5, 55), 2),
            "late_payment_fee":    random.randint(50, 2500),
            "annual_fee":          random.randint(0, 12000),
            "billing_cycle":       random.randint(10, 90),
            "minimum_payment_pct": round(random.uniform(0.5, 20), 2),
            "disclosure_provided": random.randint(0, 1),
        }

    row["label"] = is_compliant(row)
    return row


# ──────────────────────────────────────────────
# MAIN: build 300-row balanced dataset
# ──────────────────────────────────────────────
def generate_dataset(n: int = 300) -> pd.DataFrame:
    records = []

    # 40% guaranteed compliant, 40% guaranteed non-compliant, 20% random
    n_comp   = int(n * 0.40)
    n_nonc   = int(n * 0.40)
    n_rand   = n - n_comp - n_nonc

    for _ in range(n_comp):
        records.append(generate_row(force_compliant=True))
    for _ in range(n_nonc):
        records.append(generate_row(force_compliant=False))
    for _ in range(n_rand):
        records.append(generate_row(force_compliant=None))

    random.shuffle(records)
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset(300)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset shape : {df.shape}")
    print(f"Label balance :\n{df['label'].value_counts()}")
    print(df.head())
