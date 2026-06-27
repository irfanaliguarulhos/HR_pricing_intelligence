# DamoHR Pricing Intelligence: Step-by-Step Portfolio Guide

## 1. Executive Summary

DamoHR needs a governed pricing workflow for HR services. The project predicts whether a proposed deal price should be approved, reviewed, or rejected. It also explains the decision with pricing bands, statistical evidence, model metrics, monitoring checks, and a RAG-powered dashboard.

## 2. Business Objective

The business objective is margin protection without killing sales adoption. A good model should:

- Reduce underpriced deals.
- Flag risky proposals before approval.
- Keep valid deals moving quickly.
- Give sales teams explainable rate-band guidance.
- Support HR, finance, and pricing leadership with measurable governance.

## 3. Data Generation

The synthetic dataset represents HR services deals with realistic fields:

- Service line.
- Consultant seniority.
- Region.
- Client size.
- Contract length.
- Skill score.
- Utilization.
- Market rate.
- Cost rate.
- Proposed rate.
- Proposed margin.
- Win probability.
- Pricing decision target.

The generator intentionally creates messy data: duplicates, missing rates, inconsistent text values, and impossible skill scores.

Run:

```powershell
python src/damohr/generate_data.py
```

## 4. Data Preparation With PySpark

Data preparation covers:

- Duplicate removal by `deal_id`.
- Category normalization.
- Missing value imputation.
- Outlier capping for skill scores.
- Positive-rate validation.
- Clean CSV creation.

Run:

```powershell
python src/damohr/data_preparation.py
```

Interview answer:

> I use PySpark when the pricing table is too large for memory or comes from distributed data sources. The key is not just cleaning rows, but creating reproducible data quality rules that can run in production.

## 5. EDA

Core EDA questions:

- Is the target balanced across approve, review, and reject?
- Do margins separate the classes?
- Are rates right-skewed?
- Does seniority drive proposed rate?
- Are certain regions associated with higher review or reject rates?
- Does discounting reduce margin below the governance floor?

Portfolio visuals to add:

- Target class count.
- Rate distribution by seniority.
- Proposed rate versus market rate.
- Margin by decision class.
- Approval rate by region.

## 6. Statistical Testing

Implemented tests:

- Welch t-test: compares margins for approve and reject.
- Chi-square: tests association between region and pricing decision.
- ANOVA: tests rate differences across seniority levels.

Why this matters:

> Statistical tests provide evidence that business patterns are real enough to discuss, not just chart noise.

Run:

```powershell
python src/damohr/eda_stats.py
```

## 7. Econometric Testing

The project runs an OLS model on log proposed rate and uses the Breusch-Pagan test for heteroscedasticity.

Interpretation:

- If heteroscedasticity is present, rate variance increases as deal size or seniority grows.
- This supports quantile regression logic and pricing bands.
- It also supports robust validation instead of relying on one mean prediction.

Interview answer:

> In pricing, I do not trust a single average prediction. I check whether the variance structure violates OLS assumptions. If it does, I move toward quantile bands, GLMs, or tree models depending on the production objective.

## 8. Feature Engineering

Important engineered features:

- `rate_to_market_ratio`: proposed rate divided by market rate.
- `margin_gap_to_floor`: distance from governance margin floor.
- `contract_value`: rate times working days times contract months.
- `skill_seniority_index`: skill scarcity adjusted by seniority.
- `log_contract_value`: stabilized deal value.

Business logic:

> Pricing decisions are not based on raw rate alone. A high rate can be acceptable for a rare senior skill in the West, while a lower rate can be risky if margin is weak.

## 9. Feature Selection

Recommended methods:

- Mutual information for nonlinear signal.
- Permutation importance for model-agnostic ranking.
- SHAP for business-facing explanation.
- Correlation and VIF checks for econometric models.

Feature selection should balance predictive lift with explainability.

## 10. Model Building

The project trains:

- Logistic regression as interpretable baseline.
- Random forest for nonlinear interactions.
- Gradient boosting for strong tabular performance.

Primary metric:

- Macro F1, because all three classes matter.

Secondary metrics:

- Reject precision.
- Review recall.
- Confusion matrix.
- Class-level precision and recall.
- Calibration for approval confidence.

Run:

```powershell
python src/damohr/modeling.py
```

## 11. Hyperparameter Tuning

The tuned model uses `GridSearchCV` over:

- Number of trees.
- Maximum depth.
- Minimum samples per leaf.

Interview answer:

> I tune against macro F1, not just accuracy, because review and reject errors have real governance cost.

## 12. Model Selection

Model selection is based on:

- Macro F1.
- Business error cost.
- Stability across validation folds.
- Interpretability.
- Ease of deployment.
- Monitoring readiness.

Example business error framing:

- False approve: margin leakage.
- False reject: lost revenue.
- False review: slower sales cycle.

## 13. Validation

Validation checks include:

- Stratified train-test split.
- Cross-validation.
- Confusion matrix review.
- Class-level metrics.
- Bias checks by region and client size.
- Business rule backtesting.
- Manual review of false approvals and false rejects.

## 14. Deployment

The project includes a FastAPI service.

Run:

```powershell
uvicorn deployment.api:app --reload
```

Endpoints:

- `GET /health`
- `POST /predict`

Production extension:

- Docker.
- GitHub Actions.
- MLflow model registry.
- Feature store.
- Canary deployment.
- Human-in-the-loop approval queue.

## 15. Monitoring

Monitoring covers:

- PSI feature drift.
- Missing rate.
- Prediction distribution drift.
- Approval rate drift.
- Margin leakage.
- Override rate.
- Win-rate change.
- Model latency.

PSI interpretation:

- `< 0.10`: stable.
- `0.10-0.20`: watch.
- `> 0.20`: investigate.

## 16. RAG and LLM Dashboard

The dashboard includes a simple RAG-style interview assistant. A production version would use:

- Approved pricing policies.
- Model cards.
- Data dictionaries.
- Monitoring runbooks.
- Historical approval examples.
- Vector database.
- Citation-based answers.
- Guardrails against unsupported claims.

Interview answer:

> RAG is useful here because pricing users ask policy and model questions. I do not want the LLM inventing governance rules, so every answer must retrieve policy context and cite the source.

## 17. Technical Questions To Practice

### Why classification instead of regression?

Because the business workflow asks for an action: approve, review, or reject. Regression can estimate price or rate bands, but classification maps directly to governance decisions.

### Which metric matters most?

Macro F1 for balanced class performance. I would also track false approvals because they create margin risk.

### How do you handle messy data?

I define automated quality rules: duplicate checks, missing-value thresholds, categorical normalization, outlier caps, and business-valid ranges for rates and margins.

### How do you explain the model?

I explain global drivers with feature importance or SHAP, and local decisions with deal-level factors like margin gap, market-rate ratio, seniority, region, and contract value.

### How do you prove business value?

I compare margin leakage, override rate, approval cycle time, adoption, and win-rate before and after rollout. A/B testing or geo-based rollout gives stronger causal evidence.

### What econometric issue is common in pricing?

Heteroscedasticity. Senior and enterprise deals have much wider rate variance than junior or SMB deals. That weakens simple OLS assumptions.

### How do you avoid LLM hallucinations?

Use RAG, approved source documents, citation requirements, deterministic validation, policy guardrails, and human review for high-risk recommendations.

## 18. Portfolio Story

Use this narrative:

> I built DamoHR Pricing Intelligence as a full-stack data science project for HR pricing governance. It starts with messy operational pricing data, cleans it with PySpark, validates business assumptions with statistical and econometric tests, engineers pricing governance features, trains and tunes classification models, deploys the model through FastAPI, monitors drift with PSI, and exposes insights through a dashboard with a RAG knowledge base.

