import numpy as np
import pandas as pd


def population_stability_index(expected, actual, buckets=10) -> float:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 0.0001)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 0.0001)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def monitor_batch(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    checks = {}
    for col in ["proposed_rate", "market_rate", "proposed_margin", "skill_score"]:
        checks[f"psi_{col}"] = population_stability_index(reference[col], current[col])
    checks["missing_rate_current"] = float(current.isna().mean().mean())
    checks["alert"] = any(v > 0.2 for k, v in checks.items() if k.startswith("psi_"))
    return checks

