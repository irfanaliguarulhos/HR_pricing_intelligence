from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.api as sm
except Exception:
    sm = None


ROOT = Path(__file__).resolve().parents[2]


def run_statistical_tests(df: pd.DataFrame) -> dict:
    if stats is None:
        return {
            "dependency_note": "Install scipy for formal Welch t-test, chi-square, and ANOVA.",
            "margin_by_decision": df.groupby("pricing_decision")["proposed_margin"].mean().round(4).to_dict(),
            "rate_by_seniority": df.groupby("consultant_seniority")["proposed_rate"].mean().round(2).to_dict(),
        }
    approve = df.loc[df["pricing_decision"] == "approve", "proposed_margin"]
    reject = df.loc[df["pricing_decision"] == "reject", "proposed_margin"]
    contingency = pd.crosstab(df["region"], df["pricing_decision"])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    t_stat, t_p = stats.ttest_ind(approve, reject, equal_var=False)
    anova = stats.f_oneway(
        *[group["proposed_rate"].values for _, group in df.groupby("consultant_seniority")]
    )
    return {
        "welch_t_margin_approve_vs_reject": {"statistic": float(t_stat), "p_value": float(t_p)},
        "chi_square_region_vs_decision": {"statistic": float(chi2), "p_value": float(chi_p)},
        "anova_rate_by_seniority": {"statistic": float(anova.statistic), "p_value": float(anova.pvalue)},
    }


def run_econometric_checks(df: pd.DataFrame) -> dict:
    if sm is None:
        return {
            "dependency_note": "Install statsmodels for OLS and Breusch-Pagan econometric diagnostics.",
            "log_rate_variance": float(np.log(df["proposed_rate"]).var()),
            "interpretation": "Use the full environment to test heteroscedasticity formally.",
        }
    y = np.log(df["proposed_rate"])
    X = pd.get_dummies(
        df[["market_rate", "skill_score", "contract_months", "urgency_score", "region", "consultant_seniority"]],
        drop_first=True,
    )
    X = sm.add_constant(X).astype(float)
    ols = sm.OLS(y, X).fit()
    bp = sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)
    return {
        "ols_r2_log_rate": float(ols.rsquared),
        "breusch_pagan_lm_p_value": float(bp[1]),
        "interpretation": "Low BP p-value suggests heteroscedasticity, supporting tree models, quantile bands, or GLM logic.",
    }


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "damohr_pricing_clean.csv")
    print(run_statistical_tests(df))
    print(run_econometric_checks(df))
