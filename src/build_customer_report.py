import os
from collections import Counter

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PDF = os.path.join("outputs", "customer_churn_report.pdf")
EXPLANATIONS_CSV = os.path.join("outputs", "test_shap_explanations.csv")


def load_explanations(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Explanation file not found: {path}")
    return pd.read_csv(path)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("shap_")]


def summarize_feature_impact(df: pd.DataFrame, shap_cols: list[str], top_n: int = 8):
    abs_means = df[shap_cols].abs().mean().sort_values(ascending=False).head(top_n)
    signed_means = df[shap_cols].mean().reindex(abs_means.index)

    rows = []
    for col in abs_means.index:
        feature = col.replace("shap_", "")
        rows.append(
            {
                "feature": feature,
                "avg_abs_impact": float(abs_means[col]),
                "avg_direction": float(signed_means[col]),
            }
        )
    return rows


def top_at_risk_customers(df: pd.DataFrame, limit: int = 10):
    cols = [c for c in ["customerID", "pred_proba", "pred_label", "top_features", "top_shap_values"] if c in df.columns]
    if "pred_proba" not in cols:
        raise ValueError("pred_proba column is required for report generation")
    top = df.sort_values("pred_proba", ascending=False).head(limit).copy()
    return top[cols]


def build_pdf(df: pd.DataFrame, out_path: str):
    shap_cols = infer_feature_columns(df)
    if not shap_cols:
        raise ValueError("No shap_* columns found in explanation CSV")

    feature_summary = summarize_feature_impact(df, shap_cols, top_n=8)
    top_customers = top_at_risk_customers(df, limit=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#374151"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=6,
    )

    story = []
    story.append(Paragraph("Customer Churn Insight Report", title_style))
    story.append(Paragraph("Customer-facing summary based on model scores and explanation outputs", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("1) Executive Summary", subtitle_style))
    summary_text = (
        "This report highlights the strongest churn drivers identified by the model and the highest-risk customers "
        "in the test set. The goal is to help retention teams prioritize outreach based on predicted churn risk and "
        "the most influential customer attributes."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Paragraph(
        f"Total explained customers: <b>{len(df):,}</b> | Average predicted churn probability: <b>{df['pred_proba'].mean():.3f}</b>",
        body_style,
    ))

    story.append(Paragraph("2) Main Churn Drivers", subtitle_style))
    driver_rows = [["Feature", "Avg |Impact|", "Avg Direction"]]
    for row in feature_summary:
        direction = "raises risk" if row["avg_direction"] > 0 else "lowers risk"
        driver_rows.append([
            row["feature"],
            f"{row['avg_abs_impact']:.4f}",
            direction,
        ])

    driver_table = Table(driver_rows, colWidths=[2.5 * inch, 1.1 * inch, 2.4 * inch])
    driver_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f3f4f6")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(driver_table)

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("3) Highest-Risk Customers", subtitle_style))

    top_rows = [["customerID", "pred_proba", "pred_label", "top_features", "top_shap_values"]]
    for _, row in top_customers.iterrows():
        top_rows.append([
            str(row.get("customerID", "")),
            f"{float(row['pred_proba']):.3f}",
            str(int(row.get("pred_label", 0))),
            str(row.get("top_features", "")),
            str(row.get("top_shap_values", "")),
        ])

    customer_table = Table(top_rows, colWidths=[0.95 * inch, 0.75 * inch, 0.65 * inch, 2.3 * inch, 1.9 * inch])
    customer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(customer_table)

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("4) Retention Guidance", subtitle_style))
    guidance = (
        "Start with customers showing the highest predicted churn probability. Focus outreach on the top drivers "
        "that push risk upward, such as month-to-month contracts, fiber-optic service, and electronic check payments. "
        "For lower-risk customers, use automated nurture campaigns instead of direct discounting."
    )
    story.append(Paragraph(guidance, body_style))

    doc.build(story)


def main():
    df = load_explanations(EXPLANATIONS_CSV)
    build_pdf(df, OUTPUT_PDF)
    print(f"Created {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
