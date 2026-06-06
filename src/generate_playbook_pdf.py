from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import os


def build_playbook_text():
    title = "Customer Retention Playbook"

    kpis = [
        "Churn Rate: % customers lost per period.",
        "Retention Rate: 1 − Churn Rate.",
        "Customer Lifetime Value (CLV): expected revenue per customer.",
        "ARPU: Average Revenue Per User.",
        "Precision@K / Lift@K: model targeting quality.",
        "AUC / ROC: ranking performance.",
        "Cost-to-Retain vs Recoverable Revenue: campaign ROI.",
    ]

    playbook = [
        ("Goal", "Score customers by churn risk, explain drivers, and run targeted retention campaigns."),
        (
            "Segments & Actions",
            "High-risk VIPs (top 2%): 1:1 outreach + personalized offer. "
            "High-risk medium-value (top 10%): SMS/email + tailored offer. "
            "Medium-risk (next 20%): nurture content and delayed promotion."
        ),
        (
            "Offers & Rules",
            "Prefer non-monetary remedies first (service credits, expedited support). Reserve discounts for highest CLV and cap budget per period."
        ),
        (
            "Measurement",
            "Track Precision@K, Lift, recovered revenue, churn reduction, and ROI. A/B test offers."
        ),
        (
            "Ops",
            "Daily scoring job, push top-K lists to CRM, log outreach and outcomes, weekly governance review."
        ),
    ]

    return title, kpis, playbook


def create_pdf(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], alignment=1, spaceAfter=12)
    sub = ParagraphStyle("Sub", parent=styles["Heading3"], spaceAfter=6)

    title, kpis, playbook = build_playbook_text()

    elems = []
    elems.append(Paragraph(title, heading))
    elems.append(Spacer(1, 6))

    elems.append(Paragraph("Key KPIs", sub))
    for item in kpis:
        elems.append(Paragraph(f"• {item}", normal))

    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Retention Playbook", sub))
    for heading_text, body in playbook:
        elems.append(Paragraph(f"<b>{heading_text}:</b> {body}", normal))
        elems.append(Spacer(1, 4))

    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Measurement & Ops: Track outcomes, run A/B tests, and iterate weekly.", normal))

    doc.build(elems)


if __name__ == "__main__":
    out = os.path.join("outputs", "retention_playbook.pdf")
    create_pdf(out)
    print(f"Created {out}")
