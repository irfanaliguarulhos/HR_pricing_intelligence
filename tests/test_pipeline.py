from damohr.generate_data import create_synthetic_hr_pricing_data
from damohr.features import build_features


def test_data_generation_has_target_classes():
    df = create_synthetic_hr_pricing_data(1000)
    assert {"approve", "review", "reject"}.issubset(set(df["pricing_decision"]))


def test_feature_engineering_creates_governance_features():
    df = create_synthetic_hr_pricing_data(100)
    out = build_features(df)
    assert "rate_to_market_ratio" in out.columns
    assert "margin_gap_to_floor" in out.columns
    assert out["contract_value"].gt(0).all()
