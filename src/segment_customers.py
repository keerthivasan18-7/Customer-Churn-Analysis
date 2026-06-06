import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = os.path.join("outputs", "models", "rf_tuned.pkl")
FEATURES_PATH = os.path.join("outputs", "selected_features.csv")
TEST_PATH = os.path.join("data", "processed_test.csv")
EXPLANATIONS_PATH = os.path.join("outputs", "test_shap_explanations.csv")
SEGMENTS_PATH = os.path.join("outputs", "churn_risk_segments.csv")
SUMMARY_PATH = os.path.join("outputs", "churn_risk_segment_summary.csv")


def load_selected_features(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected features file not found: {path}")

    raw = pd.read_csv(path, header=None)
    values = raw.iloc[:, 0].astype(str).tolist()
    if values and values[0] == "0":
        values = values[1:]
    return values


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, list[str], object]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)
    selected = load_selected_features(FEATURES_PATH)
    return test, selected, model


def assign_risk_band(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def score_customers(test: pd.DataFrame, selected: list[str], model) -> pd.DataFrame:
    if "customerID" not in test.columns:
        raise ValueError("processed_test.csv must include customerID to export customer-level scores")

    missing = [c for c in selected if c not in test.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns in test data: {missing}")

    if "MonthlyCharges" not in test.columns:
        raise ValueError("processed_test.csv must include MonthlyCharges to compute revenue at risk")

    X = test[selected].copy()
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    revenue_at_risk_annual = test["MonthlyCharges"].astype(float).values * probs * 12
    revenue_at_risk_monthly = revenue_at_risk_annual / 12

    scored = pd.DataFrame({
        "customerID": test["customerID"].values,
        "predicted_churn_probability": probs,
        "predicted_churn_label": preds,
        "monthly_charges": test["MonthlyCharges"].astype(float).values,
        "revenue_at_risk_monthly": revenue_at_risk_monthly,
        "revenue_at_risk_annual": revenue_at_risk_annual,
    })
    scored["risk_band"] = scored["predicted_churn_probability"].apply(assign_risk_band)

    # If explanation exports exist, attach the strongest three explanation features for context.
    if os.path.exists(EXPLANATIONS_PATH):
        explain = pd.read_csv(EXPLANATIONS_PATH)
        keep_cols = [c for c in ["customerID", "top_features", "top_shap_values"] if c in explain.columns]
        if keep_cols:
            scored = scored.merge(explain[keep_cols], on="customerID", how="left")

    return scored


def build_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby("risk_band")
        .agg(
            customers=("customerID", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            median_predicted_churn_probability=("predicted_churn_probability", "median"),
            revenue_at_risk_monthly=("revenue_at_risk_monthly", "sum"),
            revenue_at_risk_annual=("revenue_at_risk_annual", "sum"),
        )
        .reindex(["High", "Medium", "Low"])
        .reset_index()
    )

    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    return summary


def main():
    os.makedirs("outputs", exist_ok=True)
    test, selected, model = load_data()
    scored = score_customers(test, selected, model)
    summary = build_summary(scored)

    scored.to_csv(SEGMENTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Wrote customer risk scores to {SEGMENTS_PATH}")
    print(f"Wrote segment summary to {SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
