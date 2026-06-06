# Customer Churn Analysis

End-to-end customer churn project built on the Telco Customer Churn Kaggle dataset.

## Project Goal

Identify customers at risk of churn, explain the drivers behind that risk, and provide a practical retention playbook that customer teams can act on.

## Business Value

- Reduce revenue leakage by retaining high-value customers before they leave.
- Focus retention spend on the customers most likely to respond.
- Surface product and service issues that drive churn.
- Improve forecasting, planning, and campaign ROI.

## Project Deliverables

- Raw and processed data pipelines
- EDA notebook: [notebooks/eda_preprocessing.ipynb](notebooks/eda_preprocessing.ipynb)
- Baseline and tuned churn models
- Customer risk segmentation and scoring
- Per-customer explanation export: `outputs/test_shap_explanations.csv`
- Risk band summary: `outputs/churn_risk_segment_summary.csv`
- Customer-facing report: `outputs/customer_churn_report.pdf`
- Retention playbook: `outputs/retention_playbook.pdf`
- Simple dashboard: `src/dashboard.py`

## Key Results

- Logistic regression (tuned): ROC AUC `0.841`, F1 `0.586`
- Random forest (tuned): ROC AUC `0.845`, F1 `0.634`
- The strongest churn drivers included tenure, monthly charges, contract type, internet service type, and payment method.

## Repository Layout

- `src/download_data.py` downloads the Kaggle dataset when credentials are available.
- `src/run_eda.py` produces the initial EDA summary.
- `src/preprocess.py` cleans and encodes the dataset.
- `src/train_models.py` trains baseline models.
- `src/tune_and_select.py` runs feature selection and tuning.
- `src/segment_customers.py` scores customers and assigns risk bands.
- `src/interpret_shap.py` generates per-customer explanations with a fallback method when SHAP is blocked.
- `src/build_customer_report.py` creates the customer-facing PDF report.

## Data Location

- Raw Kaggle file: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Processed train/test files: `data/processed_train.csv`, `data/processed_test.csv`

## Quick Start (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
python src/run_eda.py
python src/preprocess.py
python src/train_models.py
python src/tune_and_select.py
python src/segment_customers.py
python src/interpret_shap.py
python src/build_customer_report.py
streamlit run src/dashboard.py
```

## Notes

- The Kaggle download step requires a valid `kaggle.json` file or `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
- On this Windows environment, full SHAP loading may be blocked by App Control; the repository includes a fallback explanation method that still produces per-customer insights.
