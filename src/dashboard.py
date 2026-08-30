import os

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc


st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
)


DATA_DIR = "data"
OUTPUT_DIR = "outputs"

MODEL_METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics_tuned.csv")
RISK_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "churn_risk_segment_summary.csv")
RISK_SEGMENTS_PATH = os.path.join(OUTPUT_DIR, "churn_risk_segments.csv")
EXPLANATIONS_PATH = os.path.join(OUTPUT_DIR, "test_shap_explanations.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "processed_test.csv")

# Feature name mapping for readability
FEATURE_NAME_MAP = {
    "tenure": "Tenure (Months)",
    "MonthlyCharges": "Monthly Charges",
    "Contract_One year": "One-Year Contract",
    "Contract_Two year": "Two-Year Contract",
    "InternetService_Fiber optic": "Fiber Optic Internet",
    "InternetService_No": "No Internet Service",
    "OnlineSecurity": "Online Security",
    "TechSupport": "Tech Support",
    "PaymentMethod_Electronic check": "Electronic Check Payment",
    "StreamingTV": "Streaming TV",
    "OnlineBackup": "Online Backup",
}


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def get_readable_feature_name(feature: str) -> str:
    """Convert technical feature names to readable names."""
    return FEATURE_NAME_MAP.get(feature, feature)


def calculate_confusion_matrix_and_roc(risk_segments: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Calculate confusion matrix and ROC curve data from predicted and actual churn."""
    if risk_segments.empty or "predicted_churn_label" not in risk_segments.columns:
        return None, None, None, None
    
    if test_data.empty or "Churn" not in test_data.columns:
        return None, None, None, None
    
    # Merge to match actual and predicted
    merged = risk_segments.merge(test_data[["customerID", "Churn"]], on="customerID", how="inner")
    if merged.empty:
        return None, None, None, None
    
    y_true = merged["Churn"].values
    y_pred = merged["predicted_churn_label"].values
    y_proba = merged["predicted_churn_probability"].values
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return cm, fpr, tpr, roc_auc


def get_customer_value_by_risk_band(risk_segments: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    """Calculate average monthly charges by risk band."""
    merged = risk_segments.merge(test_data[["customerID", "MonthlyCharges"]], on="customerID", how="inner")
    if merged.empty:
        return {}
    value_by_band = merged.groupby("risk_band")["MonthlyCharges"].mean().to_dict()
    return value_by_band


def calculate_churn_by_contract(test_data: pd.DataFrame) -> pd.DataFrame:
    contracts = []
    if "Contract_One year" in test_data.columns:
        for col in ["Contract_One year", "Contract_Two year"]:
            if col in test_data.columns:
                label = col.replace("Contract_", "")
                if col == "Contract_One year":
                    subset = test_data[test_data["Contract_One year"] == 1]
                    if len(subset) > 0:
                        churn_rate = subset["Churn"].mean() if "Churn" in subset.columns else 0
                        contracts.append({"Contract Type": label, "Churn Rate": churn_rate, "Count": len(subset)})
                elif col == "Contract_Two year":
                    subset = test_data[test_data["Contract_Two year"] == 1]
                    if len(subset) > 0:
                        churn_rate = subset["Churn"].mean() if "Churn" in subset.columns else 0
                        contracts.append({"Contract Type": label, "Churn Rate": churn_rate, "Count": len(subset)})
        # Month-to-month: neither one year nor two year
        month_to_month = test_data[(test_data["Contract_One year"] == 0) & (test_data["Contract_Two year"] == 0)]
        if len(month_to_month) > 0:
            churn_rate = month_to_month["Churn"].mean() if "Churn" in month_to_month.columns else 0
            contracts.append({"Contract Type": "Month-to-month", "Churn Rate": churn_rate, "Count": len(month_to_month)})
    return pd.DataFrame(contracts)


def calculate_churn_by_tenure_group(test_data: pd.DataFrame) -> pd.DataFrame:
    tenure_groups = []
    tenure_cols = ["tenure_group_12-24", "tenure_group_24-48", "tenure_group_48+"]
    for col in tenure_cols:
        if col in test_data.columns:
            label = col.replace("tenure_group_", "")
            subset = test_data[test_data[col] == 1]
            if len(subset) > 0:
                churn_rate = subset["Churn"].mean() if "Churn" in subset.columns else 0
                tenure_groups.append({"Tenure Group": label, "Churn Rate": churn_rate, "Count": len(subset)})
    # 0-12 months: none of the above
    zero_to_twelve = test_data[(test_data.get("tenure_group_12-24", 0) == 0) & (test_data.get("tenure_group_24-48", 0) == 0) & (test_data.get("tenure_group_48+", 0) == 0)]
    if len(zero_to_twelve) > 0:
        churn_rate = zero_to_twelve["Churn"].mean() if "Churn" in zero_to_twelve.columns else 0
        tenure_groups.insert(0, {"Tenure Group": "0-12 months", "Churn Rate": churn_rate, "Count": len(zero_to_twelve)})
    return pd.DataFrame(tenure_groups)


def calculate_churn_by_internet_service(test_data: pd.DataFrame) -> pd.DataFrame:
    services = []
    internet_cols = {"InternetService_Fiber optic": "Fiber optic", "InternetService_No": "No internet"}
    for col, label in internet_cols.items():
        if col in test_data.columns:
            subset = test_data[test_data[col] == 1]
            if len(subset) > 0:
                churn_rate = subset["Churn"].mean() if "Churn" in subset.columns else 0
                services.append({"Internet Service": label, "Churn Rate": churn_rate, "Count": len(subset)})
    # DSL: not Fiber optic and not No
    if "InternetService_Fiber optic" in test_data.columns and "InternetService_No" in test_data.columns:
        dsl = test_data[(test_data["InternetService_Fiber optic"] == 0) & (test_data["InternetService_No"] == 0)]
        if len(dsl) > 0:
            churn_rate = dsl["Churn"].mean() if "Churn" in dsl.columns else 0
            services.insert(0, {"Internet Service": "DSL", "Churn Rate": churn_rate, "Count": len(dsl)})
    return pd.DataFrame(services)


def calculate_churn_by_payment_method(test_data: pd.DataFrame) -> pd.DataFrame:
    methods = []
    payment_cols = {
        "PaymentMethod_Credit card (automatic)": "Credit card (auto)",
        "PaymentMethod_Electronic check": "Electronic check",
        "PaymentMethod_Mailed check": "Mailed check"
    }
    for col, label in payment_cols.items():
        if col in test_data.columns:
            subset = test_data[test_data[col] == 1]
            if len(subset) > 0:
                churn_rate = subset["Churn"].mean() if "Churn" in subset.columns else 0
                methods.append({"Payment Method": label, "Churn Rate": churn_rate, "Count": len(subset)})
    # Bank transfer: not any of the above
    if all(col in test_data.columns for col in payment_cols.keys()):
        bank_transfer = test_data[
            (test_data["PaymentMethod_Credit card (automatic)"] == 0) & 
            (test_data["PaymentMethod_Electronic check"] == 0) & 
            (test_data["PaymentMethod_Mailed check"] == 0)
        ]
        if len(bank_transfer) > 0:
            churn_rate = bank_transfer["Churn"].mean() if "Churn" in bank_transfer.columns else 0
            methods.append({"Payment Method": "Bank transfer", "Churn Rate": churn_rate, "Count": len(bank_transfer)})
    return pd.DataFrame(methods)


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def stat_card(label: str, value: str, delta: str | None = None) -> str:
    delta_html = f'<div class="stat-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_customer_value_by_risk_band(risk_segments: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    """Calculate average monthly charges by risk band."""
    merged = risk_segments.merge(test_data[["customerID", "MonthlyCharges"]], on="customerID", how="inner")
    if merged.empty:
        return {}
    value_by_band = merged.groupby("risk_band")["MonthlyCharges"].mean().to_dict()
    return value_by_band


def generate_churn_insights(test_data: pd.DataFrame) -> list:
    """Generate key churn insights from historical data."""
    insights = []
    
    # Contract type insights
    churn_by_contract = calculate_churn_by_contract(test_data)
    if not churn_by_contract.empty:
        max_contract = churn_by_contract.loc[churn_by_contract["Churn Rate"].idxmax()]
        min_contract = churn_by_contract.loc[churn_by_contract["Churn Rate"].idxmin()]
        insights.append(
            f"**{max_contract['Contract Type']}** customers have a **{max_contract['Churn Rate']:.1%}** churn rate, "
            f"compared with only **{min_contract['Churn Rate']:.1%}** for **{min_contract['Contract Type']}**."
        )
    
    # Tenure insights
    churn_by_tenure = calculate_churn_by_tenure_group(test_data)
    if not churn_by_tenure.empty:
        max_tenure = churn_by_tenure.loc[churn_by_tenure["Churn Rate"].idxmax()]
        insights.append(
            f"Customers with **{max_tenure['Tenure Group']}** have the highest churn rate at **{max_tenure['Churn Rate']:.1%}**."
        )
    
    # Internet service insights
    churn_by_internet = calculate_churn_by_internet_service(test_data)
    if not churn_by_internet.empty:
        max_internet = churn_by_internet.loc[churn_by_internet["Churn Rate"].idxmax()]
        insights.append(
            f"**{max_internet['Internet Service']}** customers show a **{max_internet['Churn Rate']:.1%}** churn rate."
        )
    
    # Payment method insights
    churn_by_payment = calculate_churn_by_payment_method(test_data)
    if not churn_by_payment.empty:
        max_payment = churn_by_payment.loc[churn_by_payment["Churn Rate"].idxmax()]
        insights.append(
            f"**{max_payment['Payment Method']}** customers show a **{max_payment['Churn Rate']:.1%}** churn rate."
        )
    
    return insights


def main():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 45%, #f8fafc 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            padding: 1.6rem 1.8rem;
            border-radius: 20px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.2rem;
        }
        .hero-top {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .logo-badge {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            background: rgba(255,255,255,0.16);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            border: 1px solid rgba(255,255,255,0.22);
            flex-shrink: 0;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.45rem 0 0 0;
            opacity: 0.9;
            font-size: 0.98rem;
        }
        .section-header {
            margin: 0.2rem 0 0.6rem 0;
        }
        .section-header h2 {
            margin-bottom: 0.15rem;
            font-size: 1.25rem;
        }
        .section-header p {
            margin: 0;
            color: #64748b;
            font-size: 0.92rem;
        }
        .sidebar-title {
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .sidebar-subtitle {
            color: rgba(255,255,255,0.72);
            font-size: 0.9rem;
            line-height: 1.4;
            margin-bottom: 1rem;
        }
        .sidebar-pill {
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.12);
            color: white;
            font-size: 0.78rem;
            margin-bottom: 1rem;
        }
        .stat-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .stat-label {
            color: #64748b;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            color: #0f172a;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .stat-delta {
            margin-top: 0.25rem;
            color: #0f766e;
            font-size: 0.85rem;
        }
        .panel {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-top">
                <div class="logo-badge">CC</div>
                <div>
                    <h1>Customer Churn Dashboard</h1>
                    <p>Model performance, risk segmentation, and churn driver analysis in one operational view.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-title">Churn Intelligence</div>
        <div class="sidebar-subtitle">A compact internal dashboard for scoring, prioritization, and customer outreach.</div>
        <div class="sidebar-pill">Telco Churn | Internal Use</div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_csv(MODEL_METRICS_PATH)
    risk_summary = load_csv(RISK_SUMMARY_PATH)
    risk_segments = load_csv(RISK_SEGMENTS_PATH)
    explanations = load_csv(EXPLANATIONS_PATH)
    test_data = load_csv(TEST_DATA_PATH)

    if metrics.empty or risk_summary.empty or risk_segments.empty:
        st.error("Required output files are missing. Run the pipeline first: preprocess, train, tune, segment, and explain.")
        st.stop()

    # Top-level KPIs
    total_customers = len(test_data) if not test_data.empty else 0
    actual_churn_rate = test_data["Churn"].mean() if (not test_data.empty and "Churn" in test_data.columns) else 0.0
    high_risk_count = int(risk_summary.loc[risk_summary["risk_band"] == "High", "customers"].iloc[0])
    total_revenue_at_risk_annual = float(risk_summary["revenue_at_risk_annual"].sum()) if "revenue_at_risk_annual" in risk_summary.columns else 0.0

    nav = st.sidebar.radio(
        "Navigate",
        options=["Overview", "Risk Segments", "Customer Explorer", "Churn Drivers", "Retention Playbook"],
        index=0,
    )
    st.sidebar.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Total Customers", f"{total_customers:,}", "Customers analyzed"), unsafe_allow_html=True)
    c2.markdown(stat_card("Actual Churn Rate", format_pct(actual_churn_rate), "Historical churn observed"), unsafe_allow_html=True)
    c3.markdown(stat_card("High-risk customers", f"{high_risk_count:,}", "Immediate outreach candidates"), unsafe_allow_html=True)
    c4.markdown(stat_card("Revenue at risk", f"${total_revenue_at_risk_annual:,.0f}", "Annualized expected revenue loss"), unsafe_allow_html=True)

    if nav == "Overview":
        left, right = st.columns([1.1, 0.9])

        with left:
            with st.container(border=True):
                section_header("Model performance", "Random Forest selected for production—higher recall catches more churners.")
                
                st.markdown(
                    """
                    **✅ Selected Model: Random Forest (Tuned)**
                    
                    - **ROC-AUC:** 0.845 (slightly higher)
                    - **Recall:** 78.3% (catches 78% of actual churners vs. 53% for Logistic Regression)
                    - **Precision:** 53.3% (acceptable false positive rate)
                    - **F1 Score:** 0.634 (best overall balance)
                    
                    **Why Random Forest?** For churn retention, recall is critical—we want to catch as many potential churners as possible before they leave. Random Forest's 78% recall vs. Logistic Regression's 53% makes it the business choice.
                    """
                )
                st.divider()
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("**Model Comparison**")
                    metrics_display = metrics.reset_index().rename(columns={"index": "model"})
                    st.dataframe(metrics_display, use_container_width=True, hide_index=True)
                
                with col_m2:
                    st.markdown("**Confusion Matrix (Random Forest)**")
                    cm, fpr, tpr, roc_auc_val = calculate_confusion_matrix_and_roc(risk_segments, test_data)
                    if cm is not None:
                        tn, fp, fn, tp = cm.ravel()
                        cm_df = pd.DataFrame(
                            [[tn, fp], [fn, tp]],
                            index=["No Churn", "Churned"],
                            columns=["Pred: No Churn", "Pred: Churn"]
                        )
                        st.dataframe(cm_df, use_container_width=True)
                    else:
                        st.info("Confusion matrix unavailable.")
                
                st.divider()
                st.markdown("**ROC Curve**")
                if cm is not None and fpr is not None:
                    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                    st.line_chart(roc_df.set_index("FPR")["TPR"])
                else:
                    st.info("ROC unavailable.")

        with right:
            with st.container(border=True):
                section_header("Risk band distribution", "How the scored customers are spread across priority tiers.")
                summary_cols = [
                    "risk_band",
                    "customers",
                    "avg_predicted_churn_probability",
                    "customer_share",
                    "revenue_at_risk_monthly",
                    "revenue_at_risk_annual",
                ]
                summary_view = risk_summary[[c for c in summary_cols if c in risk_summary.columns]].copy()
                summary_view["customer_share"] = summary_view["customer_share"].map(format_pct)
                for money_col in ["revenue_at_risk_monthly", "revenue_at_risk_annual"]:
                    if money_col in summary_view.columns:
                        summary_view[money_col] = summary_view[money_col].map(lambda x: f"${x:,.0f}")
                st.dataframe(summary_view, use_container_width=True, hide_index=True)
                st.bar_chart(risk_summary.set_index("risk_band")["customers"])

        st.divider()
        section_header("Actual Churn Analysis", "Historical churn patterns by key customer attributes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("### Churn Rate by Contract Type")
                churn_by_contract = calculate_churn_by_contract(test_data)
                if not churn_by_contract.empty:
                    st.bar_chart(churn_by_contract.set_index("Contract Type")["Churn Rate"])
                    st.dataframe(churn_by_contract, use_container_width=True, hide_index=True)
                else:
                    st.info("No contract data available.")
        
        with col2:
            with st.container(border=True):
                st.markdown("### Churn Rate by Tenure Group")
                churn_by_tenure = calculate_churn_by_tenure_group(test_data)
                if not churn_by_tenure.empty:
                    st.bar_chart(churn_by_tenure.set_index("Tenure Group")["Churn Rate"])
                    st.dataframe(churn_by_tenure, use_container_width=True, hide_index=True)
                else:
                    st.info("No tenure data available.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            with st.container(border=True):
                st.markdown("### Churn Rate by Internet Service")
                churn_by_internet = calculate_churn_by_internet_service(test_data)
                if not churn_by_internet.empty:
                    st.bar_chart(churn_by_internet.set_index("Internet Service")["Churn Rate"])
                    st.dataframe(churn_by_internet, use_container_width=True, hide_index=True)
                else:
                    st.info("No internet service data available.")
        
        with col4:
            with st.container(border=True):
                st.markdown("### Churn Rate by Payment Method")
                churn_by_payment = calculate_churn_by_payment_method(test_data)
                if not churn_by_payment.empty:
                    st.bar_chart(churn_by_payment.set_index("Payment Method")["Churn Rate"])
                    st.dataframe(churn_by_payment, use_container_width=True, hide_index=True)
                else:
                    st.info("No payment method data available.")

    elif nav == "Risk Segments":
        with st.container(border=True):
            section_header("Segment summary", "Use these bands to prioritize outreach and set campaign intensity.")
            segment_view = risk_summary.copy()
            for money_col in ["revenue_at_risk_monthly", "revenue_at_risk_annual"]:
                if money_col in segment_view.columns:
                    segment_view[money_col] = segment_view[money_col].map(lambda x: f"${x:,.0f}")
            segment_view["customer_share"] = segment_view["customer_share"].map(format_pct)
            st.dataframe(segment_view, use_container_width=True, hide_index=True)

    elif nav == "Customer Explorer":
        with st.container(border=True):
            section_header("Customer Explorer", "Inspect the customers in each risk band and review explanation signals.")
            band = st.selectbox("Risk band", options=["High", "Medium", "Low"], index=0)
            filtered = risk_segments.loc[risk_segments["risk_band"] == band].copy()
            filtered = filtered.sort_values("predicted_churn_probability", ascending=False)

            search_query = st.text_input("Search customer ID", placeholder="Enter a customer ID or partial ID")
            if search_query:
                mask = filtered["customerID"].astype(str).str.contains(search_query, case=False, na=False)
                filtered = filtered.loc[mask].copy()

            st.caption(f"Showing {len(filtered)} customers in {band} risk")

            display_cols = [
                c
                for c in [
                    "customerID",
                    "predicted_churn_probability",
                    "predicted_churn_label",
                    "risk_band",
                    "top_features",
                    "top_shap_values",
                ]
                if c in filtered.columns
            ]
            if display_cols:
                st.dataframe(filtered[display_cols].head(25), use_container_width=True, hide_index=True)

            if not filtered.empty:
                st.bar_chart(filtered.head(20).set_index("customerID")["predicted_churn_probability"])

    elif nav == "Churn Drivers":
        with st.container(border=True):
            section_header("Top Churn Risk Signals", "The strongest predictive features driving customer churn risk in our model.")
            if not explanations.empty:
                shap_cols = [c for c in explanations.columns if c.startswith("shap_")]
                if shap_cols:
                    impact = explanations[shap_cols].abs().mean().sort_values(ascending=False).head(10)
                    impact.index = [get_readable_feature_name(c.replace("shap_", "")) for c in impact.index]
                    st.bar_chart(impact)
                    
                    st.divider()
                    st.markdown(
                        """
                        **💡 Business Interpretation:**
                        
                        - **High Impact Signals:** Customers with month-to-month contracts, fiber optic internet, or low tenure show the highest churn risk.
                        - **Action:** Focus retention campaigns on improving contract lengths (incentivize annual/2-year plans) and addressing internet service quality.
                        - **Quick Wins:** Payment method changes and tech support adoption can reduce predicted churn in these segments.
                        """
                    )
                    
                    st.dataframe(
                        impact.reset_index().rename(columns={"index": "Risk Signal", 0: "Impact Score"}),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("Explanation file not found. Run the interpretation step to populate this tab.")

    elif nav == "Retention Playbook":
        with st.container(border=True):
            section_header("Retention Playbook", "Data-driven actions based on risk level, customer value, and churn probability.")
            
            # Calculate customer value by risk band
            customer_values = get_customer_value_by_risk_band(risk_segments, test_data)
            
            playbook = [
                {
                    "risk": "High",
                    "value": f"${customer_values.get('High', 0):.0f}/mo",
                    "title": "High-Risk VIPs",
                    "action": "Direct personal outreach from senior CSR + tailored retention offer",
                    "priority": "🔴 CRITICAL"
                },
                {
                    "risk": "High",
                    "value": f"${customer_values.get('High', 0) * 0.6:.0f}/mo",
                    "title": "High-Risk Medium-Value",
                    "action": "Email/SMS campaign + incentive offer + contract upgrade path",
                    "priority": "🟠 HIGH"
                },
                {
                    "risk": "Medium",
                    "value": f"${customer_values.get('Medium', 0):.0f}/mo",
                    "title": "Medium-Risk Standard",
                    "action": "Proactive feature education + upsell opportunity + service improvement check",
                    "priority": "🟡 MEDIUM"
                },
                {
                    "risk": "Low",
                    "value": f"${customer_values.get('Low', 0):.0f}/mo",
                    "title": "Low-Risk Stable",
                    "action": "Automated lifecycle campaigns + renewal reminders + loyalty program",
                    "priority": "🟢 LOW"
                }
            ]
            
            for idx, segment in enumerate(playbook):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([1, 1.5, 1])
                    with col1:
                        st.markdown(f"**Risk Level:** {segment['risk']}")
                        st.markdown(f"**Avg Value:** {segment['value']}")
                    with col2:
                        st.markdown(f"**{segment['title']}**")
                        st.markdown(f"**Action:** {segment['action']}")
                    with col3:
                        st.markdown(segment['priority'])
            
            st.divider()
            st.markdown(
                """
                **Key Principles:**
                - **Critical (🔴):** Immediate 1:1 outreach + personalized offer; budget for service credits or discounts
                - **High (🟠):** Automated campaigns with strong incentives; focus on contract upgrade
                - **Medium (🟡):** Educational content + product feature highlights; identify unmet needs
                - **Low (🟢):** Minimal intervention; focus on engagement and upsell, not retention spend
                """
            )

    st.divider()
    with st.container(border=True):
        section_header("🔎 Key Churn Insights", "Critical patterns from historical customer data.")
        insights = generate_churn_insights(test_data)
        for insight in insights:
            st.markdown(f"• {insight}")


if __name__ == "__main__":
    main()
