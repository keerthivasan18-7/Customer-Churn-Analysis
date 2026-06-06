import os
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(in_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv", out_dir="data"):
    df = pd.read_csv(in_path)

    # Keep customerID separately so we can attach it to train/test splits
    customer_ids = None
    if "customerID" in df.columns:
        customer_ids = df["customerID"].copy()

    # Fix TotalCharges (stored as string)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # If TotalCharges is missing, fill with MonthlyCharges * tenure (reasonable proxy)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    # Work on a features copy without customerID
    features = df.drop(columns=["customerID"]) if "customerID" in df.columns else df.copy()

    # Normalize 'No internet service' / 'No phone service' to 'No'
    features = features.replace({"No internet service": "No", "No phone service": "No"})

    # Binary mappings
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for c in binary_cols:
        if c in features.columns:
            features[c] = features[c].map({"Yes": 1, "No": 0})

    # Target
    if "Churn" in features.columns:
        features["Churn"] = features["Churn"].map({"Yes": 1, "No": 0})

    # Tenure bins
    features["tenure_group"] = pd.cut(features["tenure"], bins=[0, 12, 24, 48, 999], labels=["0-12", "12-24", "24-48", "48+"])

    # Categorical columns to one-hot
    cat_cols = [c for c in ["gender", "InternetService", "Contract", "PaymentMethod", "tenure_group"] if c in features.columns]
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True)

    # Reorder so target is last
    if "Churn" in features.columns:
        cols = [c for c in features.columns if c != "Churn"] + ["Churn"]
        features = features[cols]

    os.makedirs(out_dir, exist_ok=True)
    processed_path = os.path.join(out_dir, "processed_telco.csv")
    # Save the full processed table (without customerID)
    features.to_csv(processed_path, index=False)

    # Train/test split (preserve customerID mapping)
    if "Churn" in features.columns:
        # Use the same indices to split so we can map customerIDs back
        train_df, test_df = train_test_split(features, test_size=0.2, random_state=42, stratify=features["Churn"]) 

        # Attach customerID columns back to train/test using original indices
        if customer_ids is not None:
            train_ids = customer_ids.loc[train_df.index]
            test_ids = customer_ids.loc[test_df.index]
            train_out = train_df.copy()
            test_out = test_df.copy()
            train_out.insert(0, "customerID", train_ids.values)
            test_out.insert(0, "customerID", test_ids.values)
            train_out.to_csv(os.path.join(out_dir, "processed_train.csv"), index=False)
            test_out.to_csv(os.path.join(out_dir, "processed_test.csv"), index=False)
            # Also save id-only files
            train_ids.to_csv(os.path.join(out_dir, "processed_train_ids.csv"), index=False, header=["customerID"]) 
            test_ids.to_csv(os.path.join(out_dir, "processed_test_ids.csv"), index=False, header=["customerID"]) 
        else:
            train_df.to_csv(os.path.join(out_dir, "processed_train.csv"), index=False)
            test_df.to_csv(os.path.join(out_dir, "processed_test.csv"), index=False)

    print(f"Processed data written to {processed_path}")
    if "Churn" in features.columns:
        print(f"Train/test written to {out_dir}/processed_train.csv and {out_dir}/processed_test.csv")


if __name__ == "__main__":
    csv_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        print(f"Input CSV not found at {csv_path}. Run download_data.py first.")
    else:
        preprocess(csv_path, out_dir="data")
