import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import joblib


def load_data(dirpath="data"):
    train = pd.read_csv(os.path.join(dirpath, "processed_train.csv"))
    test = pd.read_csv(os.path.join(dirpath, "processed_test.csv"))
    return train, test


def prepare_xy(df):
    drop_cols = ["Churn"]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    X = df.drop(columns=drop_cols).copy()
    y = df["Churn"].copy()
    return X, y


def feature_selection_by_rf(X, y, top_k=20):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    k = min(top_k, len(imp))
    selected = imp.head(k).index.tolist()
    return selected, imp


def evaluate(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
    }


def main():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    train, test = load_data("data")
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    # Feature selection
    selected, importances = feature_selection_by_rf(X_train, y_train, top_k=20)
    importances.to_csv(os.path.join("outputs", "feature_importances.csv"))
    print("Selected features:", selected)

    # Reduce datasets
    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Logistic Regression with class_weight tuning
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, solver="saga"))])
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 5, 10],
        "clf__penalty": ["l2", "l1"],
        "clf__class_weight": [None, "balanced"],
    }
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train_sel, y_train)
    best_log = gs.best_estimator_
    print("Best logistic params:", gs.best_params_)

    # Random Forest tuning
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    }
    rs = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42)
    rs.fit(X_train_sel, y_train)
    best_rf = rs.best_estimator_
    print("Best RF params:", rs.best_params_)

    # Evaluate on test set
    res_log = evaluate(best_log, X_test_sel, y_test)
    res_rf = evaluate(best_rf, X_test_sel, y_test)
    df_results = pd.DataFrame([res_log, res_rf], index=["logistic_tuned", "rf_tuned"]) 
    df_results.to_csv(os.path.join("outputs", "model_metrics_tuned.csv"))

    # Save models and selected features
    joblib.dump(best_log, os.path.join("outputs", "models", "logistic_tuned.pkl"))
    joblib.dump(best_rf, os.path.join("outputs", "models", "rf_tuned.pkl"))
    pd.Series(selected).to_csv(os.path.join("outputs", "selected_features.csv"), index=False)

    print(df_results)


if __name__ == "__main__":
    main()
