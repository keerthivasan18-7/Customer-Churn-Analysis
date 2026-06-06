# Customer Churn Analysis - Presentation Outline

## Slide 1: Title
- Customer Churn Analysis
- Telco Customer Churn dataset
- Your name / date

## Slide 2: Business Problem
- Customers are leaving before their lifetime value is fully realized
- Retaining existing customers is cheaper than acquiring new ones
- Goal: predict churn and prioritize intervention

## Slide 3: Project Goal
- Build a churn prediction model
- Explain the drivers of churn
- Turn model output into a retention playbook

## Slide 4: Dataset Overview
- Source: Kaggle Telco Customer Churn
- Rows: 7,043
- Target: Churn (Yes/No)
- Key fields: tenure, contract, charges, internet service, payment method

## Slide 5: EDA Highlights
- Churn is imbalanced
- Month-to-month contracts show higher churn
- Higher monthly charges correlate with higher churn
- Short tenure customers are at higher risk

## Slide 6: Preprocessing
- Converted TotalCharges to numeric
- Imputed missing TotalCharges using MonthlyCharges x tenure
- Encoded binary features
- One-hot encoded key categorical variables
- Added tenure groups

## Slide 7: Modeling Approach
- Baseline: Logistic Regression
- Baseline: Random Forest
- Improved with feature selection, class weighting, and hyperparameter tuning

## Slide 8: Model Results
- Tuned Logistic Regression: ROC AUC 0.841, F1 0.586
- Tuned Random Forest: ROC AUC 0.845, F1 0.634
- Random Forest improved recall and overall ranking quality

## Slide 9: Interpretability
- Main drivers: tenure, monthly charges, contract type, internet service, payment method
- Per-customer explanation export created for targeted outreach
- Fallback contribution method used where SHAP was blocked

## Slide 10: Customer Segmentation & Risk Scoring
- High-risk: 309 customers, average churn probability 0.815
- Medium-risk: 339 customers, average churn probability 0.556
- Low-risk: 761 customers, average churn probability 0.151
- Use risk bands to prioritize outreach and budget

## Slide 11: Retention Playbook
- High-risk VIPs: direct outreach
- Medium-risk customers: targeted email/SMS offers
- Product-driven churn: route to support/product teams
- Lower-risk customers: nurture campaigns

## Slide 12: Business Impact
- Reduce churn and protect recurring revenue
- Improve campaign ROI through targeting
- Surface root causes of dissatisfaction
- Support planning and forecasting

## Slide 13: Next Steps
- Deploy scoring pipeline
- Monitor model drift and campaign performance
- A/B test retention offers
- Extend to dashboard or CRM integration
