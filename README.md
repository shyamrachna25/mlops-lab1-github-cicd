# MLOps Lab 1 — GitHub Fundamentals & CI/CD

## My changes

Replaced `calculator.py` with `src/data_validation.py` — five data validation
and feature engineering functions instead of arithmetic, since these are
more representative of real MLOps data pipelines:

- `check_missing_values` — flags columns exceeding a missing-value threshold
- `check_value_range` — checks a numeric column stays within expected bounds
- `check_duplicate_rows` — counts duplicate rows
- `validate_schema` — validates a DataFrame's columns/dtypes against expectations
- `normalize_column` — min-max normalization

Both `pytest` and `unittest` suites (12 tests each, all passing) were rewritten
to match, and both GitHub Actions workflows run them automatically on every
push to `main`.

## Running it

\`\`\`bash
pip install -r requirements.txt
pytest test/test_pytest.py -v
python3 -m unittest test.test_unittest -v
\`\`\`
