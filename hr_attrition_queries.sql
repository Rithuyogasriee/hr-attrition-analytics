-- =============================================================================
-- HR ATTRITION ANALYTICS — SQL QUERY BANK
-- Database: hr_analytics  |  Table: employee_attrition
-- Compatible with: MySQL 8+ / PostgreSQL 13+ / SQLite 3+
-- =============================================================================

-- ── TABLE CREATION ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS employee_attrition (
    EmployeeID              INT PRIMARY KEY AUTO_INCREMENT,
    Age                     INT,
    Attrition               VARCHAR(3),   -- 'Yes' / 'No'
    BusinessTravel          VARCHAR(30),
    Department              VARCHAR(40),
    DistanceFromHome        INT,
    Education               INT,          -- 1=Below College … 5=Doctor
    EnvironmentSatisfaction INT,          -- 1=Low … 4=Very High
    Gender                  VARCHAR(10),
    JobRole                 VARCHAR(40),
    JobSatisfaction         INT,
    MaritalStatus           VARCHAR(15),
    MonthlyIncome           INT,
    NumCompaniesWorked      INT,
    OverTime                VARCHAR(3),
    PerformanceRating       INT,
    StockOptionLevel        INT,
    TotalWorkingYears       INT,
    TrainingTimesLastYear   INT,
    WorkLifeBalance         INT,
    YearsAtCompany          INT,
    YearsInCurrentRole      INT
);

-- =============================================================================
-- QUERY 1 — OVERALL ATTRITION RATE
-- Business Use: KPI card for Executive Dashboard
-- =============================================================================
SELECT
    COUNT(*)                                                       AS TotalEmployees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS TotalAttrition,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                                              AS AttritionRate_Pct
FROM employee_attrition;

-- =============================================================================
-- QUERY 2 — ATTRITION BY DEPARTMENT
-- Business Use: Identify which business unit needs immediate HR intervention
-- =============================================================================
SELECT
    Department,
    COUNT(*)                                                       AS HeadCount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS Attrited,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct
FROM employee_attrition
GROUP BY Department
ORDER BY AttritionRate_Pct DESC;

-- =============================================================================
-- QUERY 3 — ATTRITION BY JOB ROLE (RANKED)
-- Business Use: Prioritise retention spend by role criticality
-- =============================================================================
SELECT
    JobRole,
    COUNT(*)                                                       AS HeadCount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS Attrited,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct,
    ROUND(AVG(MonthlyIncome), 0)                                   AS AvgMonthlyIncome,
    ROUND(AVG(JobSatisfaction), 2)                                 AS AvgJobSatisfaction
FROM employee_attrition
GROUP BY JobRole
ORDER BY AttritionRate_Pct DESC;

-- =============================================================================
-- QUERY 4 — OVERTIME IMPACT ON ATTRITION
-- Business Use: Quantify the cost of over-scheduling (HR policy insight)
-- =============================================================================
SELECT
    OverTime,
    COUNT(*)                                                       AS TotalEmployees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS Attrited,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct
FROM employee_attrition
GROUP BY OverTime
ORDER BY AttritionRate_Pct DESC;

-- =============================================================================
-- QUERY 5 — INCOME BRACKETS vs ATTRITION
-- Business Use: Determine salary thresholds that significantly reduce churn
-- =============================================================================
SELECT
    CASE
        WHEN MonthlyIncome < 3000  THEN '< ₹3,000'
        WHEN MonthlyIncome < 6000  THEN '₹3,000–6,000'
        WHEN MonthlyIncome < 10000 THEN '₹6,000–10,000'
        ELSE '> ₹10,000'
    END                                                            AS IncomeBracket,
    COUNT(*)                                                       AS HeadCount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS Attrited,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct
FROM employee_attrition
GROUP BY IncomeBracket
ORDER BY AttritionRate_Pct DESC;

-- =============================================================================
-- QUERY 6 — SATISFACTION SCORECARD (Multi-dimensional)
-- Business Use: Identify the satisfaction dimension driving attrition most
-- =============================================================================
SELECT
    ROUND(AVG(CASE WHEN Attrition='Yes' THEN JobSatisfaction END), 2)         AS LeaveAvgJobSat,
    ROUND(AVG(CASE WHEN Attrition='No'  THEN JobSatisfaction END), 2)         AS StayAvgJobSat,
    ROUND(AVG(CASE WHEN Attrition='Yes' THEN EnvironmentSatisfaction END), 2) AS LeaveAvgEnvSat,
    ROUND(AVG(CASE WHEN Attrition='No'  THEN EnvironmentSatisfaction END), 2) AS StayAvgEnvSat,
    ROUND(AVG(CASE WHEN Attrition='Yes' THEN WorkLifeBalance END), 2)         AS LeaveAvgWLB,
    ROUND(AVG(CASE WHEN Attrition='No'  THEN WorkLifeBalance END), 2)         AS StayAvgWLB
FROM employee_attrition;

-- =============================================================================
-- QUERY 7 — TENURE COHORT ANALYSIS
-- Business Use: New hire retention risk (first 2 years are most critical)
-- =============================================================================
SELECT
    CASE
        WHEN YearsAtCompany <= 2  THEN '0–2 Years'
        WHEN YearsAtCompany <= 5  THEN '3–5 Years'
        WHEN YearsAtCompany <= 10 THEN '6–10 Years'
        ELSE '10+ Years'
    END                                                            AS TenureBand,
    COUNT(*)                                                       AS HeadCount,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct,
    ROUND(AVG(MonthlyIncome), 0)                                   AS AvgIncome
FROM employee_attrition
GROUP BY TenureBand
ORDER BY
    CASE TenureBand
        WHEN '0–2 Years'  THEN 1
        WHEN '3–5 Years'  THEN 2
        WHEN '6–10 Years' THEN 3
        ELSE 4
    END;

-- =============================================================================
-- QUERY 8 — TOP 10 HIGH-RISK EMPLOYEES (for HR Manager View)
-- Business Use: Proactive retention — flag employees to engage now
-- =============================================================================
SELECT
    EmployeeID,
    Age,
    JobRole,
    Department,
    MonthlyIncome,
    JobSatisfaction,
    WorkLifeBalance,
    OverTime,
    YearsAtCompany,
    -- Custom risk score (weightings based on model feature importance)
    ROUND(
        (5 - JobSatisfaction) * 0.30
        + (5 - WorkLifeBalance) * 0.20
        + CASE WHEN OverTime = 'Yes' THEN 0.25 ELSE 0 END
        + (CASE WHEN YearsAtCompany <= 2 THEN 0.15 ELSE 0 END)
        + (CASE WHEN MonthlyIncome < 4000 THEN 0.10 ELSE 0 END)
    , 2)                                                           AS RiskScore
FROM employee_attrition
WHERE Attrition = 'No'   -- Currently employed
ORDER BY RiskScore DESC
LIMIT 10;

-- =============================================================================
-- QUERY 9 — ATTRITION COST ESTIMATE
-- Business Use: Quantify monetary impact (cost = 50–200% of annual salary)
-- =============================================================================
SELECT
    Department,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS EmployeesLost,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 0) AS AvgMonthlyIncome,
    -- Conservative estimate: replacement cost = 6 months salary
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome * 6 ELSE 0 END)
    , 0)                                                           AS EstimatedReplacementCost
FROM employee_attrition
GROUP BY Department
ORDER BY EstimatedReplacementCost DESC;

-- =============================================================================
-- QUERY 10 — TRAINING ROI vs ATTRITION
-- Business Use: Does more training reduce attrition? Policy decision support
-- =============================================================================
SELECT
    TrainingTimesLastYear,
    COUNT(*)                                                       AS HeadCount,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct
FROM employee_attrition
GROUP BY TrainingTimesLastYear
ORDER BY TrainingTimesLastYear;

-- =============================================================================
-- VIEW — HR ATTRITION SUMMARY (reusable in Power BI / Tableau)
-- =============================================================================
CREATE OR REPLACE VIEW vw_AttritionSummary AS
SELECT
    Department,
    JobRole,
    Gender,
    MaritalStatus,
    OverTime,
    COUNT(*)                                                       AS HeadCount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)            AS AttritionCount,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 1
    )                                                              AS AttritionRate_Pct,
    ROUND(AVG(MonthlyIncome), 0)                                   AS AvgMonthlyIncome,
    ROUND(AVG(JobSatisfaction), 2)                                 AS AvgJobSatisfaction,
    ROUND(AVG(WorkLifeBalance), 2)                                 AS AvgWorkLifeBalance
FROM employee_attrition
GROUP BY Department, JobRole, Gender, MaritalStatus, OverTime;
