import os
import pandas as pd


def main():
    csv_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please run src/download_data.py or place the CSV in data/.")
        return

    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nColumns and dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    if "Churn" in df.columns:
        print("\nTarget distribution (Churn):")
        print(df["Churn"].value_counts(dropna=False))
    else:
        print("\nWarning: 'Churn' column not found in dataset.")

    summary = df.describe(include="all").transpose()
    os.makedirs("outputs", exist_ok=True)
    summary.to_csv(os.path.join("outputs", "eda_summary.csv"))
    print("\nEDA summary saved to outputs/eda_summary.csv")


if __name__ == "__main__":
    main()
