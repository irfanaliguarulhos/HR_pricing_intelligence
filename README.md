# DamoHR Pricing Intelligence Classification Project

End-to-end portfolio project for an HR pricing analytics team. The project predicts whether a proposed HR services deal price should be `approve`, `review`, or `reject`, and explains the decision with rate-band governance, model metrics, statistical tests, monitoring, and a small RAG knowledge base for interview-style Q&A.

## Business Problem

DamoHR sells HR staffing, recruiting, payroll, and workforce analytics services. Sales teams propose daily rates for engagements. Finance and HR leadership need a governed pricing workflow that protects margin while preserving win-rate.

The classification target is:

- `approve`: proposed price is inside the defensible pricing band.
- `review`: price is close to a guardrail and needs manager approval.
- `reject`: price is outside governance rules or margin risk is too high.

## Project Coverage

- Synthetic HR pricing data creation with realistic messy data issues.
- PySpark data preparation and data quality checks.
- EDA and statistical testing.
- Econometric diagnostics.
- Feature engineering and feature selection.
- Classification model building and model selection.
- Hyperparameter tuning.
- Validation and testing.
- Deployment with FastAPI.
- Monitoring with drift metrics and model health checks.
- Streamlit dashboard.
- RAG knowledge base for technical interview questions.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/damohr/generate_data.py
python src/damohr/run_pipeline.py
pytest
streamlit run dashboard/app.py
uvicorn deployment.api:app --reload
```

If PySpark is not available locally, the pipeline falls back to pandas for the portfolio demo path.

## Main Files

- `docs/STEP_BY_STEP_PROJECT_GUIDE.md`: full learning and interview walkthrough.
- `src/damohr/generate_data.py`: synthetic DamoHR pricing data.
- `src/damohr/data_preparation.py`: PySpark/pandas cleaning.
- `src/damohr/eda_stats.py`: EDA, statistical tests, econometric checks.
- `src/damohr/features.py`: feature engineering and selection.
- `src/damohr/modeling.py`: models, tuning, validation, metrics.
- `src/damohr/monitoring.py`: drift and production health.
- `deployment/api.py`: prediction API using `models/pricing_classifier.pkl`.
- `dashboard/app.py`: LLM/RAG-enabled dashboard shell.
- `rag/knowledge_base.md`: interview knowledge base.
