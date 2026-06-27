from pathlib import Path
import pickle

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception:
    ColumnTransformer = None

from damohr.features import TARGET, build_features


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def train_models(df: pd.DataFrame) -> dict:
    if ColumnTransformer is None:
        return train_rule_based_demo_model(df)
    df = build_features(df)
    X = df.drop(columns=[TARGET, "deal_id"])
    y = df[TARGET]
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    candidates = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    results = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "pipeline": pipe,
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "report": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    tuned = tune_best_model(preprocess, X_train, y_train)
    tuned_preds = tuned.predict(X_test)
    results["tuned_random_forest"] = {
        "pipeline": tuned,
        "macro_f1": f1_score(y_test, tuned_preds, average="macro"),
        "report": classification_report(y_test, tuned_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, tuned_preds).tolist(),
    }
    final_name = max(results, key=lambda k: results[k]["macro_f1"])
    MODEL_DIR.mkdir(exist_ok=True)
    save_model(results[final_name]["pipeline"], MODEL_DIR / "pricing_classifier.pkl")
    return {"best_baseline": best_name, "final_model": final_name, "results": results}


class RuleBasedPricingModel:
    classes_ = ["approve", "reject", "review"]

    def predict(self, X):
        decisions = []
        for _, row in X.iterrows():
            margin = (row["proposed_rate"] - row["cost_rate"]) / row["proposed_rate"]
            ratio = row["proposed_rate"] / row["market_rate"]
            if 0.88 <= ratio <= 1.18 and margin >= 0.28:
                decisions.append("approve")
            elif 0.82 <= ratio <= 1.27 and margin >= 0.20:
                decisions.append("review")
            else:
                decisions.append("reject")
        return decisions

    def predict_proba(self, X):
        preds = self.predict(X)
        rows = []
        for pred in preds:
            rows.append([0.9 if cls == pred else 0.05 for cls in self.classes_])
        return rows


def train_rule_based_demo_model(df: pd.DataFrame) -> dict:
    df = build_features(df)
    X = df.drop(columns=[TARGET, "deal_id"])
    y = df[TARGET].tolist()
    model = RuleBasedPricingModel()
    preds = model.predict(X)
    accuracy = sum(a == b for a, b in zip(y, preds)) / len(y)
    MODEL_DIR.mkdir(exist_ok=True)
    save_model(model, MODEL_DIR / "pricing_classifier.pkl")
    return {
        "best_baseline": "rule_based_governance",
        "final_model": "rule_based_governance",
        "results": {
            "rule_based_governance": {
                "accuracy": accuracy,
                "note": "Install scikit-learn to run logistic regression, random forest, gradient boosting, and tuning.",
            }
        },
    }


def tune_best_model(preprocess, X_train, y_train):
    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    grid = {
        "model__n_estimators": [120, 220],
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [1, 4],
    }
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_


def save_model(model, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    data = pd.read_csv(ROOT / "data" / "processed" / "damohr_pricing_clean.csv")
    summary = train_models(data)
    print(summary["final_model"], summary["results"][summary["final_model"]]["macro_f1"])
