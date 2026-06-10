# =============================================================================
# HR Employee Attrition Analysis & Prediction
# Author: [Your Name] | MBA Business Analytics & HR
# Dataset: IBM HR Analytics Employee Attrition Dataset (Public Domain)
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── STYLE SETTINGS ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = ["#2E86AB", "#E84855", "#F6AE2D", "#27AE60", "#8E44AD"]
sns.set_palette(PALETTE)

print("=" * 60)
print("  HR ATTRITION ANALYTICS — FULL PIPELINE")
print("=" * 60)

# =============================================================================
# STEP 1 — GENERATE / LOAD DATASET
# The IBM HR Analytics dataset is widely available on Kaggle (public domain).
# Here we recreate it programmatically so the repo is self-contained.
# To use the real CSV: df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# =============================================================================

np.random.seed(42)
n = 1470  # matches IBM dataset size

def generate_hr_data(n):
    """Generate a synthetic dataset that mirrors IBM HR Analytics structure."""
    age            = np.random.randint(18, 60, n)
    dept           = np.random.choice(
        ["Sales", "Research & Development", "Human Resources"],
        n, p=[0.30, 0.60, 0.10]
    )
    job_role       = np.random.choice(
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative", "Manager",
         "Sales Representative", "Research Director", "Human Resources"],
        n
    )
    gender         = np.random.choice(["Male", "Female"], n, p=[0.60, 0.40])
    education      = np.random.randint(1, 6, n)          # 1=Below College … 5=Doctor
    marital_status = np.random.choice(
        ["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22]
    )
    monthly_income = (np.random.lognormal(8.5, 0.5, n)).clip(1009, 20000).astype(int)
    job_satisfaction  = np.random.randint(1, 5, n)
    env_satisfaction  = np.random.randint(1, 5, n)
    work_life_balance = np.random.randint(1, 5, n)
    overtime       = np.random.choice(["Yes", "No"], n, p=[0.28, 0.72])
    years_at_co    = np.random.randint(0, 40, n)
    years_in_role  = np.clip(np.random.randint(0, 18, n), 0, years_at_co)
    num_companies  = np.random.randint(0, 10, n)
    distance_home  = np.random.randint(1, 30, n)
    training_times = np.random.randint(0, 7, n)
    perf_rating    = np.random.choice([3, 4], n, p=[0.85, 0.15])
    stock_option   = np.random.randint(0, 4, n)
    business_travel= np.random.choice(
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        n, p=[0.19, 0.71, 0.10]
    )

    # Attrition probability — injecting realistic HR drivers
    logit = (
        -1.5
        + 0.8   * (age < 30).astype(int)            # younger → higher churn
        - 0.00008 * monthly_income                   # higher pay → lower churn
        + 1.2   * (overtime == "Yes").astype(int)    # overtime → higher churn
        - 0.5   * job_satisfaction
        - 0.4   * work_life_balance
        + 0.6   * (marital_status == "Single").astype(int)
        + 0.5   * (business_travel == "Travel_Frequently").astype(int)
        - 0.08  * years_at_co
        + 0.15  * distance_home / 10
    )
    prob_attr = 1 / (1 + np.exp(-logit))
    attrition = (np.random.rand(n) < prob_attr).astype(int)

    return pd.DataFrame({
        "Age": age, "Attrition": np.where(attrition, "Yes", "No"),
        "BusinessTravel": business_travel, "Department": dept,
        "DistanceFromHome": distance_home, "Education": education,
        "EnvironmentSatisfaction": env_satisfaction, "Gender": gender,
        "JobRole": job_role, "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies, "OverTime": overtime,
        "PerformanceRating": perf_rating, "StockOptionLevel": stock_option,
        "TotalWorkingYears": years_at_co, "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_co,
        "YearsInCurrentRole": years_in_role,
    })

df = generate_hr_data(n)
df.to_csv("data/hr_attrition_raw.csv", index=False)
print(f"\n[1] Dataset created → {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3))

# =============================================================================
# STEP 2 — DATA CLEANING
# =============================================================================
print("\n[2] DATA CLEANING")
print("-" * 40)

# 2a. Check for nulls
null_report = df.isnull().sum()
print("Null values per column:\n", null_report[null_report > 0] if null_report.sum() > 0 else "  None found ✓")

# 2b. Check duplicates
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
df.drop_duplicates(inplace=True)

# 2c. Outlier detection on MonthlyIncome using IQR
Q1, Q3 = df["MonthlyIncome"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df["MonthlyIncome"] < Q1 - 1.5 * IQR) | (df["MonthlyIncome"] > Q3 + 1.5 * IQR)]
print(f"Income outliers detected: {len(outliers)} rows (retained — business-valid)")

# 2d. Derived features
df["AttritionFlag"]       = (df["Attrition"] == "Yes").astype(int)
df["IncomePerYear"]       = (df["MonthlyIncome"] * 12).astype(int)
df["SeniorityBand"]       = pd.cut(df["YearsAtCompany"],
                                    bins=[-1, 2, 5, 10, 100],
                                    labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
df["AgeGroup"]            = pd.cut(df["Age"],
                                    bins=[17, 25, 35, 45, 100],
                                    labels=["18-25", "26-35", "36-45", "46+"])
df["SatisfactionScore"]   = (df["JobSatisfaction"] + df["EnvironmentSatisfaction"]
                              + df["WorkLifeBalance"]) / 3

df.to_csv("data/hr_attrition_clean.csv", index=False)
print(f"\nCleaned dataset saved → {df.shape[0]} rows × {df.shape[1]} columns")

# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n[3] EDA — KEY STATISTICS")
print("-" * 40)

attr_rate = df["AttritionFlag"].mean() * 100
print(f"Overall Attrition Rate  : {attr_rate:.1f}%")

dept_attr = df.groupby("Department")["AttritionFlag"].mean().mul(100).round(1)
print("\nAttrition by Department:\n", dept_attr)

role_attr = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False).round(1)
print("\nTop 5 Roles by Attrition:\n", role_attr.head())

# ── FIG 1: Overview Dashboard ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("HR Attrition — Executive Overview Dashboard", fontsize=16, fontweight="bold", y=1.01)

# 1a Attrition split
counts = df["Attrition"].value_counts()
axes[0, 0].pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#E84855", "#2E86AB"], startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0, 0].set_title("Overall Attrition Split")

# 1b Attrition by Department
dept_counts = df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)
dept_counts.plot(kind="bar", ax=axes[0, 1], color=["#2E86AB", "#E84855"],
                 edgecolor="white", rot=20)
axes[0, 1].set_title("Attrition Count by Department")
axes[0, 1].set_xlabel("")
axes[0, 1].legend(title="Attrition")

# 1c Age distribution by attrition
df.groupby("Attrition")["Age"].plot(kind="hist", alpha=0.6, bins=20,
                                     ax=axes[0, 2], edgecolor="white")
axes[0, 2].set_title("Age Distribution by Attrition")
axes[0, 2].set_xlabel("Age")
axes[0, 2].legend(["No Attrition", "Attrition"])

# 1d Monthly Income boxplot
df.boxplot(column="MonthlyIncome", by="Attrition", ax=axes[1, 0],
           patch_artist=True,
           boxprops=dict(facecolor="#2E86AB", color="navy"),
           medianprops=dict(color="orange", linewidth=2))
axes[1, 0].set_title("Monthly Income vs Attrition")
axes[1, 0].set_xlabel("Attrition")
plt.sca(axes[1, 0])
plt.title("Monthly Income vs Attrition")

# 1e Overtime impact
ot = df.groupby(["OverTime", "Attrition"]).size().unstack(fill_value=0)
ot_pct = ot.div(ot.sum(axis=1), axis=0) * 100
ot_pct.plot(kind="bar", ax=axes[1, 1], color=["#2E86AB", "#E84855"],
            edgecolor="white", rot=0)
axes[1, 1].set_title("Attrition Rate — OverTime vs Non-OverTime")
axes[1, 1].set_xlabel("Overtime")
axes[1, 1].set_ylabel("% Employees")
axes[1, 1].legend(title="Attrition")

# 1f Job Satisfaction
sat = df.groupby(["JobSatisfaction", "Attrition"]).size().unstack(fill_value=0)
sat_pct = sat.div(sat.sum(axis=1), axis=0) * 100
sat_pct["Yes"].plot(kind="bar", ax=axes[1, 2], color="#E84855", edgecolor="white", rot=0)
axes[1, 2].set_title("Attrition Rate by Job Satisfaction Level")
axes[1, 2].set_xlabel("Job Satisfaction (1=Low, 4=High)")
axes[1, 2].set_ylabel("Attrition Rate (%)")

plt.tight_layout()
plt.savefig("visualizations/01_overview_dashboard.png", bbox_inches="tight")
plt.close()
print("\n[VIZ] Saved → visualizations/01_overview_dashboard.png")

# ── FIG 2: Heatmap — Correlation Matrix ───────────────────────────────────────
numeric_cols = ["Age", "MonthlyIncome", "JobSatisfaction", "EnvironmentSatisfaction",
                "WorkLifeBalance", "YearsAtCompany", "NumCompaniesWorked",
                "TrainingTimesLastYear", "DistanceFromHome", "AttritionFlag"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Correlation Matrix — HR Attrition Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("visualizations/02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("[VIZ] Saved → visualizations/02_correlation_heatmap.png")

# ── FIG 3: Attrition by Seniority Band & Marital Status ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sen = df.groupby(["SeniorityBand", "Attrition"]).size().unstack(fill_value=0)
sen_pct = sen.div(sen.sum(axis=1), axis=0) * 100
sen_pct["Yes"].plot(kind="bar", ax=axes[0], color=PALETTE[:4], edgecolor="white", rot=0)
axes[0].set_title("Attrition Rate by Tenure Band")
axes[0].set_xlabel("Years at Company")
axes[0].set_ylabel("Attrition Rate (%)")
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

mar = df.groupby(["MaritalStatus", "Attrition"]).size().unstack(fill_value=0)
mar_pct = mar.div(mar.sum(axis=1), axis=0) * 100
mar_pct["Yes"].plot(kind="bar", ax=axes[1], color=PALETTE[1:4], edgecolor="white", rot=0)
axes[1].set_title("Attrition Rate by Marital Status")
axes[1].set_xlabel("Marital Status")
axes[1].set_ylabel("Attrition Rate (%)")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("Demographic Attrition Patterns", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("visualizations/03_demographic_patterns.png", bbox_inches="tight")
plt.close()
print("[VIZ] Saved → visualizations/03_demographic_patterns.png")

# ── FIG 4: Income Distribution by Role ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
role_income = df.groupby("JobRole")["MonthlyIncome"].median().sort_values()
colors = ["#E84855" if df[df["JobRole"] == role]["AttritionFlag"].mean() > 0.20
          else "#2E86AB" for role in role_income.index]
bars = ax.barh(role_income.index, role_income.values, color=colors, edgecolor="white")
ax.set_title("Median Monthly Income by Job Role\n(Red = >20% Attrition Risk)", fontsize=13, fontweight="bold")
ax.set_xlabel("Median Monthly Income (₹)")
for bar, val in zip(bars, role_income.values):
    ax.text(val + 100, bar.get_y() + bar.get_height() / 2,
            f"₹{val:,.0f}", va="center", fontsize=9)
legend_patches = [
    mpatches.Patch(color="#E84855", label="High Attrition Risk (>20%)"),
    mpatches.Patch(color="#2E86AB", label="Normal Attrition Risk"),
]
ax.legend(handles=legend_patches, loc="lower right")
plt.tight_layout()
plt.savefig("visualizations/04_income_by_role.png", bbox_inches="tight")
plt.close()
print("[VIZ] Saved → visualizations/04_income_by_role.png")

# =============================================================================
# STEP 4 — MACHINE LEARNING MODEL
# =============================================================================
print("\n[4] MACHINE LEARNING — ATTRITION PREDICTION")
print("-" * 40)

# 4a. Feature Engineering
le = LabelEncoder()
cat_cols = ["BusinessTravel", "Department", "Gender", "JobRole", "MaritalStatus", "OverTime"]
df_ml = df.copy()
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

feature_cols = [
    "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "Gender", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked", "OverTime",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "SatisfactionScore"
]

X = df_ml[feature_cols]
y = df_ml["AttritionFlag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 4b. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_prob = lr.predict_proba(X_test_sc)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_prob)
print(f"\nLogistic Regression AUC  : {lr_auc:.3f}")
print(classification_report(y_test, lr_pred, target_names=["Stay", "Leave"]))

# 4c. Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                             class_weight="balanced")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_auc  = roc_auc_score(y_test, rf_prob)
print(f"Random Forest AUC        : {rf_auc:.3f}")
print(classification_report(y_test, rf_pred, target_names=["Stay", "Leave"]))

# 4d. Feature Importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

# ── FIG 5: ML Results Dashboard ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Machine Learning — Attrition Prediction Results", fontsize=14, fontweight="bold")

# ROC curves
for model_prob, model_name, color in [
    (lr_prob, f"Logistic Reg (AUC={lr_auc:.2f})", "#2E86AB"),
    (rf_prob, f"Random Forest (AUC={rf_auc:.2f})", "#E84855"),
]:
    fpr, tpr, _ = roc_curve(y_test, model_prob)
    axes[0].plot(fpr, tpr, label=model_name, color=color, lw=2)
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Confusion Matrix (Random Forest)
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Random Forest — Confusion Matrix")

# Feature Importance
top10 = importances.tail(10)
top10.plot(kind="barh", ax=axes[2], color="#2E86AB", edgecolor="white")
axes[2].set_title("Top 10 Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("visualizations/05_ml_results.png", bbox_inches="tight")
plt.close()
print("[VIZ] Saved → visualizations/05_ml_results.png")

# =============================================================================
# STEP 5 — BUSINESS INSIGHTS SUMMARY REPORT
# =============================================================================
print("\n[5] BUSINESS INSIGHTS")
print("=" * 60)

insights = {
    "Overall Attrition Rate": f"{attr_rate:.1f}%",
    "Highest Risk Department": dept_attr.idxmax(),
    "Top Risk Job Role": role_attr.idxmax(),
    "Overtime Attrition Premium": f"{df[df['OverTime']=='Yes']['AttritionFlag'].mean()*100:.1f}% vs "
                                   f"{df[df['OverTime']=='No']['AttritionFlag'].mean()*100:.1f}%",
    "Low-Satisfaction Attrition": f"{df[df['JobSatisfaction']==1]['AttritionFlag'].mean()*100:.1f}%",
    "Best ML Model": f"Random Forest (AUC: {rf_auc:.3f})",
    "Top Predictor": importances.tail(1).index[0],
}

for k, v in insights.items():
    print(f"  ► {k:<38} : {v}")

# Save insights to CSV for report
pd.DataFrame(insights.items(), columns=["Insight", "Value"]).to_csv(
    "reports/business_insights.csv", index=False
)
print("\n[DONE] All outputs saved. Check /visualizations and /reports folders.")
