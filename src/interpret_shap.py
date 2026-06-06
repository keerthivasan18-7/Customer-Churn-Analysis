import os
import pandas as pd
import numpy as np
import joblib

def main():
    model_path = os.path.join("outputs", "models", "rf_tuned.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run tuning first.")

    model = joblib.load(model_path)

    # Load selected features and test data
    sel_path = os.path.join("outputs", "selected_features.csv")
    if not os.path.exists(sel_path):
        raise FileNotFoundError("selected_features.csv not found in outputs/")
    # Read selected features robustly whether header exists or not
    try:
        df_sel = pd.read_csv(sel_path, header=None)
        selected = df_sel.iloc[:, 0].astype(str).tolist()
        # If the first entry looks like the header name '0', drop it
        if len(selected) > 0 and selected[0].strip() == '0':
            selected = selected[1:]
    except Exception:
        df_sel = pd.read_csv(sel_path)
        selected = df_sel.iloc[:, 0].astype(str).tolist()

    test_path = os.path.join("data", "processed_test.csv")
    raw_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    test = pd.read_csv(test_path)
    raw = pd.read_csv(raw_path)

    # Ensure customerID column exists in test
    if "customerID" not in test.columns:
        raise ValueError("processed_test.csv must include customerID column")

    X_test = test[selected].copy()

    # Try SHAP; if environment blocks native extensions, fall back to baseline replacement method
    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)

    shap_df = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        # For binary classification, shap_vals is list [class0, class1]
        if isinstance(shap_vals, list):
            shap_class1 = shap_vals[1]
        else:
            shap_class1 = shap_vals
        shap_df = pd.DataFrame(shap_class1, columns=selected)
    except Exception as e:
        print("SHAP import or explainer failed, falling back to baseline-replacement contributions. Error:", e)
        # Load train medians for baseline replacement
        train_path = os.path.join("data", "processed_train.csv")
        train = pd.read_csv(train_path)
        # If customerID present, drop it for medians
        if "customerID" in train.columns:
            train_vals = train.drop(columns=["customerID"])
        else:
            train_vals = train
        medians = train_vals[selected].median()

        # For each feature, replace with median and compute delta in predicted probability
        base_probs = probs
        contribs = np.zeros((X_test.shape[0], len(selected)))
        for i, feat in enumerate(selected):
            X_temp = X_test.copy()
            X_temp[feat] = medians[feat]
            p_temp = model.predict_proba(X_temp)[:,1]
            # contribution = original probability - probability with feature set to baseline
            contribs[:, i] = base_probs - p_temp
        shap_df = pd.DataFrame(contribs, columns=selected)
    out = test[["customerID"]].reset_index(drop=True)
    out["pred_proba"] = probs
    out["pred_label"] = preds

    # Add top contributors
    abs_shap = np.abs(shap_df.values)
    topk = 3
    top_feats = []
    top_vals = []
    shap_array = shap_df.values
    for row in range(abs_shap.shape[0]):
        idxs = np.argsort(-abs_shap[row])[:topk]
        feats = [selected[i] for i in idxs]
        vals = [shap_array[row, i] for i in idxs]
        top_feats.append(";".join(feats))
        top_vals.append(";".join([f"{v:.4f}" for v in vals]))

    out["top_features"] = top_feats
    out["top_shap_values"] = top_vals

    # Append full shap values per feature (prefix shap_)
    shap_df.columns = [f"shap_{c}" for c in shap_df.columns]
    result = pd.concat([out, shap_df], axis=1)

    os.makedirs("outputs", exist_ok=True)
    result.to_csv(os.path.join("outputs", "test_shap_explanations.csv"), index=False)
    print(f"Wrote explanations to outputs/test_shap_explanations.csv")


if __name__ == "__main__":
    main()
