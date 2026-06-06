import os

import pandas as pd
import streamlit as st


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


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


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


RETENTION_PLAYBOOK = [
    {
        "title": "High-risk VIPs",
        "subtitle": "Top-priority customers with the largest expected revenue loss.",
        "actions": [
            "Direct outreach from a senior customer success rep.",
            "Offer a tailored retention incentive or service credit.",
            "Schedule a follow-up within 3 business days.",
        ],
    },
    {
        "title": "High-risk medium-value",
        "subtitle": "Customers who are likely to churn and still justify targeted retention spend.",
        "actions": [
            "Send an email or SMS with a personalized offer.",
            "Highlight plan upgrades or contract savings.",
            "Track response within 7 days and re-score if needed.",
        ],
    },
    {
        "title": "Product-driven churn",
        "subtitle": "Customers whose explanation signals suggest service or product dissatisfaction.",
        "actions": [
            "Route the account to support or product teams.",
            "Prioritize issue resolution over discounts.",
            "Capture feedback to identify recurring pain points.",
        ],
    },
    {
        "title": "Low-risk nurture",
        "subtitle": "Stable customers who do not need costly intervention.",
        "actions": [
            "Use automated lifecycle campaigns.",
            "Share product education and value reminders.",
            "Avoid unnecessary discounting.",
        ],
    },
]


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

    if metrics.empty or risk_summary.empty or risk_segments.empty:
        st.error("Required output files are missing. Run the pipeline first: preprocess, train, tune, segment, and explain.")
        st.stop()

    # Top-level KPIs
    best_model_row = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    high_risk_share = float(risk_summary.loc[risk_summary["risk_band"] == "High", "customer_share"].iloc[0])
    high_risk_count = int(risk_summary.loc[risk_summary["risk_band"] == "High", "customers"].iloc[0])
    avg_high_risk = float(risk_summary.loc[risk_summary["risk_band"] == "High", "avg_predicted_churn_probability"].iloc[0])
    total_revenue_at_risk_annual = float(risk_summary["revenue_at_risk_annual"].sum()) if "revenue_at_risk_annual" in risk_summary.columns else 0.0

    nav = st.sidebar.radio(
        "Navigate",
        options=["Overview", "Risk Segments", "Customer Explorer", "Churn Drivers", "Retention Playbook"],
        index=0,
    )
    st.sidebar.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Best ROC AUC", f"{best_model_row['roc_auc']:.3f}", f"Top model: {best_model_row.name}"), unsafe_allow_html=True)
    c2.markdown(stat_card("High-risk customers", f"{high_risk_count:,}", "Immediate outreach candidates"), unsafe_allow_html=True)
    c3.markdown(stat_card("High-risk share", format_pct(high_risk_share), "Share of scored customers"), unsafe_allow_html=True)
    c4.markdown(stat_card("Revenue at risk", f"${total_revenue_at_risk_annual:,.0f}", "Annualized expected revenue loss"), unsafe_allow_html=True)

    if nav == "Overview":
        left, right = st.columns([1.1, 0.9])

        with left:
            with st.container(border=True):
                section_header("Model performance", "Compare the tuned models on the held-out test set.")
                metrics_display = metrics.reset_index().rename(columns={"index": "model"})
                st.dataframe(metrics_display, use_container_width=True, hide_index=True)
                st.bar_chart(metrics.set_index(metrics.index)[["roc_auc", "f1", "precision", "recall"]])

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
            section_header("Top churn drivers", "The strongest model signals behind churn risk.")
            if not explanations.empty:
                shap_cols = [c for c in explanations.columns if c.startswith("shap_")]
                if shap_cols:
                    impact = explanations[shap_cols].abs().mean().sort_values(ascending=False).head(10)
                    impact.index = [c.replace("shap_", "") for c in impact.index]
                    st.bar_chart(impact)
                    st.dataframe(
                        impact.reset_index().rename(columns={"index": "feature", 0: "avg_abs_impact"}),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("Explanation file not found. Run the interpretation step to populate this tab.")

    elif nav == "Retention Playbook":
        with st.container(border=True):
            section_header("Retention Playbook", "Action recommendations tied to risk bands and churn signals.")
            st.write(
                "Use this page to translate churn scores into retention actions. Focus expensive interventions on the highest-value customers first."
            )

            cols = st.columns(2)
            for idx, item in enumerate(RETENTION_PLAYBOOK):
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="panel">
                            <h3 style="margin-top:0;margin-bottom:0.2rem;">{item['title']}</h3>
                            <p style="margin-top:0;color:#64748b;">{item['subtitle']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for action in item["actions"]:
                        st.markdown(f"- {action}")

    st.divider()
    with st.container(border=True):
        section_header("Operational note", "A lightweight internal dashboard that can be extended later.")
        st.write(
            "This dashboard is ready to run locally and can be extended with authentication, CRM integration, or deployment."
        )


if __name__ == "__main__":
    main()
