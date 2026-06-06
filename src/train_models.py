import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


def load_data(dirpath="data"):
    train_path = os.path.join(dirpath, "processed_train.csv")
    test_path = os.path.join(dirpath, "processed_test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def prepare_xy(df):
    if "Churn" not in df.columns:
        raise ValueError("Churn column not found")
    drop_cols = ["Churn"]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    X = df.drop(columns=drop_cols).copy()
    y = df["Churn"].copy()
    return X, y


def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def main():
    os.makedirs("outputs/models", exist_ok=True)
    train, test = load_data("data")
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    # Ensure columns in test match train (they should)
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    # Models
    models = {}

    logreg = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    models["logistic_regression"] = logreg

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # Evaluate
    results = []
    for name, m in models.items():
        res = evaluate_model(m, X_test, y_test)
        res["model"] = name
        results.append(res)
        # Save model
        joblib.dump(m, os.path.join("outputs", "models", f"{name}.pkl"))

    results_df = pd.DataFrame(results).set_index("model")
    results_df.to_csv(os.path.join("outputs", "model_metrics.csv"))
    print(results_df)


if __name__ == "__main__":
    main()
