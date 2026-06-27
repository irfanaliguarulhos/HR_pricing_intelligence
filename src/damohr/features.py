import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
except Exception:
    SelectKBest = None
    LabelEncoder = None


TARGET = "pricing_decision"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["proposed_rate", "market_rate", "cost_rate", "proposed_margin", "skill_score"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    out["rate_to_market_ratio"] = out["proposed_rate"] / out["market_rate"]
    out["margin_gap_to_floor"] = out["proposed_margin"] - 0.28
    out["contract_value"] = out["proposed_rate"] * 21 * out["contract_months"]
    out["skill_seniority_index"] = out["skill_score"] * out["consultant_seniority"].map(
        {"Junior": 0.8, "Mid": 1.0, "Senior": 1.25, "Principal": 1.5}
    )
    out["log_contract_value"] = np.log1p(out["contract_value"])
    return out


def split_xy(df: pd.DataFrame):
    if LabelEncoder is None:
        X = df.drop(columns=[TARGET, "deal_id"])
        return X, df[TARGET]
    X = df.drop(columns=[TARGET, "deal_id"])
    y = LabelEncoder().fit_transform(df[TARGET])
    return X, y


def select_features(preprocessed_X, y, feature_names, k=18):
    if SelectKBest is None:
        return feature_names[:k], None
    selector = SelectKBest(mutual_info_classif, k=min(k, len(feature_names)))
    selector.fit(preprocessed_X, y)
    selected = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
    return selected, selector
