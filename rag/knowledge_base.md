# DamoHR Pricing Intelligence RAG Knowledge Base

## Business Framing

The pricing model supports HR services governance. It classifies deals as approve, review, or reject based on proposed rate, market rate, cost, margin, contract duration, region, seniority, skill scarcity, and win probability.

## Data Preparation

Important data issues include duplicate deal IDs, missing proposed rates, inconsistent text categories, impossible skill scores, outlier rates, and margin calculation errors. PySpark is used for scalable cleaning; pandas fallback supports local demos.

## EDA

Key checks include target balance, rate distribution by seniority, margin by decision class, region and client-size effects, and proposed-rate-to-market-rate ratio.

## Statistical Tests

Welch t-test compares approve versus reject margins. Chi-square tests whether region and pricing decision are independent. ANOVA tests whether proposed rates differ by seniority. These tests help translate model patterns into business evidence.

## Econometrics

OLS on log proposed rate gives an interpretable baseline. Breusch-Pagan detects heteroscedasticity. Heteroscedasticity supports quantile bands, Gamma GLM logic, or tree models because HR pricing variance increases with seniority and market rate.

## Feature Engineering

Important features include rate-to-market ratio, margin gap to floor, contract value, skill-seniority index, utilization, urgency, discount percent, region, service line, client size, and contract months.

## Feature Selection

Mutual information, model importance, permutation importance, and SHAP can identify which variables contribute most to pricing decisions. In pricing governance, feature selection must also consider business explainability.

## Model Building

Baseline models include logistic regression, random forest, and gradient boosting. The primary classification metric is macro F1 because approve, review, and reject must all perform well. Secondary metrics include precision for reject, recall for review, confusion matrix, and calibration.

## Hyperparameter Tuning

GridSearchCV tunes random forest depth, estimators, and minimum leaf size using macro F1. In production, randomized search or Bayesian optimization can reduce compute time.

## Validation

Validation includes stratified train-test split, cross-validation, confusion matrix, class-level precision and recall, business rule backtesting, reject false-negative review, and bias checks across region/client size.

## Deployment

FastAPI exposes `/health` and `/predict`. The model artifact is loaded from `models/pricing_classifier.joblib`. Docker, CI/CD, and a model registry such as MLflow can be added for production.

## Monitoring

Monitoring includes PSI for feature drift, missing-value rate, class distribution drift, prediction confidence, approval-rate changes, and business KPIs such as override rate, margin protection, adoption, and win-rate.

## RAG and LLM Dashboard

RAG prevents hallucination by grounding answers in approved pricing policy, model documentation, statistical test results, and monitoring runbooks. The dashboard retrieves relevant snippets from this knowledge base for interview and stakeholder explanations.

