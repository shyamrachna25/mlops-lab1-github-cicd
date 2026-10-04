# mlops-lab1-github-cicd

Modification from original lab: Replaced calculator.py with data_validation.py, containing data validation functions (missing-value checks, range checks, duplicate detection, schema validation) and a feature engineering function (min-max normalization) — functions more directly relevant to MLOps data pipelines than the original arithmetic example. Both pytest and unittest suites were rewritten to match, and both GitHub Actions workflows were adapted accordingly.
