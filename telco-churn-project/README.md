# IBM Telco Customer Churn EDA Project

Author: `yanyixuan`

This project studies the IBM Telco Customer Churn sample and asks which customer segments should be prioritized for retention experiments.

## Contents

- `IBM_Telco_Churn_EDA_Report.pdf`: final project report for submission.
- `analysis.py`: reproducible cleaning, EDA, feature engineering, and hypothesis tests.
- `build_report.py`: builds the report from the generated results and figures.
- `build_cleaning_evidence.py`: creates the before-and-after cleaning and encoding evidence figure.
- `figures/`: report-ready visualizations.
- `results/`: schema, missingness, descriptive statistics, contingency tables, tests, and a full JSON summary.
- `data/`: processed analysis outputs. The original IBM sample is referenced by URL and can be downloaded by `analysis.py`.

## Reproduce

```powershell
$env:PYTHONPATH = "tmp/churn_packages"
python output/telco_churn_project/analysis.py
python output/telco_churn_project/build_report.py
```

The source dataset is the IBM public sample:

https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

The analysis uses a discovery sample and a held-out confirmation sample with `random_state=42`. The three planned tests use Bonferroni correction with family alpha 0.05.

## Main result

Month-to-month customers showed a 43.7% churn rate in the confirmation sample, compared with 6.0% for one-year or two-year customers. The risk difference was 37.7 percentage points (95% CI 34.5-40.9), and the association remained significant after correction.

The result is associative. It supports a retention experiment; it does not prove that changing contract type alone causes lower churn.
