# Streamlit Visual App (Tasks 1–8)

## Run locally
From the repository root (the folder that contains `python_project/`):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r python_project/requirements.txt
streamlit run python_project/streamlit_app.py
```

## What’s inside
- **Tasks 1–3:** LP scheduling (min cost, fairer schedule, skills coverage)
- **Tasks 4–5:** A&E sample summaries + rate breakdowns
- **Task 6:** Logistic regression breach prediction (threshold tuned on validation for target recall)
- **Task 7:** CRUD-style A&E data management (writes an audit trail)
- **Task 8:** Audit trail viewer + action frequency chart

If your environment can’t find HiGHS, install `highspy` (included in requirements) or configure PuLP to use another solver.
