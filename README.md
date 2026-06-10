#  HR Employee Attrition Analytics & Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![SQL](https://img.shields.io/badge/SQL-Query%20Bank-lightgrey?logo=mysql)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**An end-to-end HR analytics project 
I built to understand why employees leave companies, using Python, combining data cleaning, EDA, machine learning and business recommendations — built to demonstrate analytical thinking for Business Analyst and HR Analyst.

</div>

---

##  Business Problem Statement

A mid-sized IT company is experiencing **16.1% annual employee attrition** — significantly above the industry benchmark of 10–12%. This translates to an estimated **₹2.3 Cr in annual replacement costs** (recruiting, onboarding, lost productivity).

HR leadership needs answers to three questions:
1. **Who** is most likely to leave in the next 6 months?
2. **Why** are employees leaving — what are the root drivers?
3. **What** actions can HR and business managers take right now?

---

## Project Structure

```
hr-attrition-analytics/
│
├── data/
│   ├── hr_attrition_raw.csv          # Raw dataset (1,470 employee records)
│   └── hr_attrition_clean.csv        # Cleaned + feature-engineered dataset
│
├── notebooks/
│   └── HR_Attrition_Analysis.ipynb   # Full Jupyter notebook (step-by-step)
│
├── src/
│   └── analysis.py                   # Production-ready Python script
│
├── sql/
│   └── hr_attrition_queries.sql      # 10 SQL queries + reusable VIEW
│
├── visualizations/
│   ├── 01_overview_dashboard.png     # Executive overview (6-panel)
│   ├── 02_correlation_heatmap.png    # Variable correlation matrix
│   ├── 03_demographic_patterns.png   # Tenure & marital status analysis
│   ├── 04_income_by_role.png         # Income vs risk by job role
│   └── 05_ml_results.png             # ROC curves + confusion matrix + feature importance
│
├── reports/
│   └── business_insights.csv         # Key metrics summary
│
├── docs/
│   └── powerbi_dashboard_guide.md    # Power BI setup instructions
│
├── README.md
└── requirements.txt
```

---

## Dataset Description

| Field | Description |
|---|---|
| **Source** | IBM HR Analytics Employee Attrition Dataset (Public Domain) |
| **Records** | 1,470 employees |
| **Features** | 21 columns — demographics, compensation, satisfaction scores, tenure |
| **Target** | `Attrition` — Yes (left) / No (stayed) |
| **Class Balance** | ~16% Attrition (Yes) / ~84% No Attrition |

**Key Variables:**

| Variable | Type | Notes |
|---|---|---|
| Age | Numeric | 18–60 years |
| MonthlyIncome | Numeric | ₹1,009–₹20,000 |
| JobSatisfaction | Ordinal | 1 (Low) → 4 (Very High) |
| OverTime | Binary | Yes / No |
| YearsAtCompany | Numeric | Tenure in current company |
| BusinessTravel | Categorical | Non-Travel / Travel_Rarely / Travel_Frequently |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/hr-attrition-analytics.git
cd hr-attrition-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full analysis
python src/analysis.py

# 4. Or open the notebook
jupyter notebook notebooks/HR_Attrition_Analysis.ipynb
```
# 5. Improvement — SMOTE Applied

**Problem Identified:**
Initial model had severe class imbalance — 1,439 employees (Stay) vs only 31 employees (Leave). Model was biased towards predicting everyone stays.

**Solution:**
Applied SMOTE (Synthetic Minority Oversampling Technique) exclusively on training data to avoid data leakage.

| | Before SMOTE | After SMOTE |
|---|---|---|
| Stay (0) | 1,439 | 1,151 |
| Leave (1) | 31 | 1,151 |
| Balance | Imbalanced | Perfectly Balanced |

**Key Learnings:**
- SMOTE applied only on training set — never on test data (prevents data leakage)
- OverTime confirmed as the strongest attrition predictor after rebalancing
- Recall for attrition class improved significantly

---

## Data Cleaning Steps

| Step | Action | Outcome |
|---|---|---|
| Null Check | `df.isnull().sum()` | Zero nulls confirmed |
| Duplicate Removal | `df.drop_duplicates()` | Clean unique records |
| Outlier Detection | IQR method on MonthlyIncome | Flagged, retained (business-valid) |
| Feature Engineering | Seniority bands, Age groups, Satisfaction composite score | 5 new derived features |
| Encoding | LabelEncoder for ML pipeline | All categoricals converted |

---

## Key Visualizations

| Chart | Insight Delivered |
|---|---|
| `01_overview_dashboard.png` | 6-panel executive summary — attrition split, dept breakdown, income, overtime, satisfaction |
| `02_correlation_heatmap.png` | Variable relationships — MonthlyIncome & JobSatisfaction most correlated with retention |
| `03_demographic_patterns.png` | 0–2 year employees and single employees show highest attrition |
| `04_income_by_role.png` | Roles earning <₹4,000/month carry >20% attrition risk |
| `05_ml_results.png` | ROC curves, confusion matrix, top 10 predictors by importance |

---

##  Machine Learning Model

### Models Compared

| Model | AUC Score | Notes |
|---|---|---|
| Logistic Regression | ~0.76 | Interpretable baseline |
| **Random Forest** | **~0.82** | **Best performer — deployed** |

### Top Predictors (Feature Importance)

1. **MonthlyIncome** — Strongest retention lever
2.  **OverTime** — Working overtime doubles attrition risk
3.  **JobSatisfaction** — 1-point drop = significant churn increase
4. **WorkLifeBalance** — Critical for employee wellbeing
5. **YearsAtCompany** — Early-tenure employees most at risk

---

##  Business Insights

> These are the actionable findings derived from the analysis.

| # | Insight | Impact |
|---|---|---|
| 1 | Employees doing overtime are **2.3× more likely** to leave | High — affects 28% of workforce |
| 2 | New joiners (0–2 years) show **highest attrition rate** (~22%) | High — new hire failure is costly |
| 3 | Salary below ₹4,000/month is the clearest attrition trigger | High — compensation benchmarking needed |
| 4 | Single employees attrite at nearly **2× the rate** of married peers | Medium — review engagement programs |
| 5 | Frequent business travel adds **30% more attrition risk** | Medium — travel policy review recommended |
| 6 | Training frequency shows an inverse U-shape — 2–3 sessions optimal | Low-Medium — training ROI can be optimized |

---

## Recommendations

**Immediate (0–3 months):**
Implement mandatory overtime caps (max 10 hours/week) with manager alerts
Launch a 30-60-90 day onboarding pulse survey for all new joiners
Salary benchmarking audit for roles with >20% attrition rate

**Short-Term (3–6 months):**
Deploy the ML model as an HR dashboard widget to flag at-risk employees monthly
Introduce flexible work-from-home policy for frequent travelers
Establish a stay-interview program (not just exit interviews)

**Strategic (6–12 months):**
 Build a predictive attrition score into the HRIS system
 Create a structured career-path framework for high-risk roles (Sales Rep, Lab Technician)
 Revise stock option & compensation policy for employees in the 0–3 year tenure band

---

## Power BI Dashboard

A 3-page Power BI dashboard was built on this dataset:

- **Page 1:** Executive Summary — KPI cards, attrition rate trend, department breakdown
- **Page 2:** Workforce Deep-Dive — income distribution, satisfaction scores, tenure analysis
- **Page 3:** Predictive View — at-risk employees table, ML risk score, recommended actions

>  See `docs/powerbi_dashboard_guide.md` for setup steps

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.9+ | Data analysis & ML pipeline |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
| Scikit-Learn | ML models (Logistic Regression, Random Forest) |
| SQL (MySQL/SQLite) | Querying structured HR data |
| Power BI | Interactive business dashboard |
| Jupyter Notebook | Exploratory analysis |

---

## About the Author

Rithuyogasriee 
MBA — Business Analytics & Human Resources | Class of 2026  
📍 Chennai, India

> Aspiring Business / HR / Data Analyst with hands-on experience in Python and Power BI.  
> Passionate about using data to solve real HR and business problems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).  
Dataset adapted from the IBM HR Analytics dataset (Public Domain / CC0).

---

*If this project helped you, please give it a star!*
