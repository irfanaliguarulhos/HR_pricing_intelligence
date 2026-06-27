from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def create_synthetic_hr_pricing_data(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    service = rng.choice(
        ["Recruiting", "Payroll", "HR Analytics", "Benefits Admin", "Workforce Planning"],
        n_rows,
        p=[0.30, 0.22, 0.18, 0.17, 0.13],
    )
    seniority = rng.choice(["Junior", "Mid", "Senior", "Principal"], n_rows, p=[0.25, 0.35, 0.30, 0.10])
    region = rng.choice(["Northeast", "Midwest", "South", "West", "Remote"], n_rows)
    client_size = rng.choice(["SMB", "MidMarket", "Enterprise"], n_rows, p=[0.35, 0.40, 0.25])
    contract_months = rng.integers(1, 25, n_rows)
    skills = rng.normal(65, 15, n_rows).clip(5, 100)
    urgency = rng.integers(1, 6, n_rows)
    utilization = rng.normal(0.78, 0.12, n_rows).clip(0.35, 0.99)

    service_mult = pd.Series(service).map(
        {"Recruiting": 1.00, "Payroll": 0.88, "HR Analytics": 1.28, "Benefits Admin": 0.95, "Workforce Planning": 1.18}
    ).to_numpy()
    seniority_mult = pd.Series(seniority).map({"Junior": 0.70, "Mid": 1.0, "Senior": 1.35, "Principal": 1.75}).to_numpy()
    region_mult = pd.Series(region).map({"Northeast": 1.12, "Midwest": 0.92, "South": 0.95, "West": 1.25, "Remote": 1.02}).to_numpy()
    size_mult = pd.Series(client_size).map({"SMB": 0.92, "MidMarket": 1.0, "Enterprise": 1.14}).to_numpy()

    market_rate = 520 * service_mult * seniority_mult * region_mult * size_mult * (1 + skills / 350)
    cost_rate = market_rate * rng.normal(0.62, 0.07, n_rows).clip(0.42, 0.82)
    proposed_rate = market_rate * rng.normal(1.0, 0.16, n_rows) * (1 + (urgency - 3) * 0.025)
    proposed_margin = (proposed_rate - cost_rate) / proposed_rate
    discount_pct = rng.beta(2, 10, n_rows) * 0.30
    win_probability = 1 / (1 + np.exp((proposed_rate / market_rate - 1.03) * 6 - 0.6))

    floor = market_rate * 0.88
    ceiling = market_rate * 1.18
    status = np.where(
        (proposed_rate >= floor) & (proposed_rate <= ceiling) & (proposed_margin >= 0.28),
        "approve",
        np.where((proposed_margin >= 0.20) & (proposed_rate >= floor * 0.94) & (proposed_rate <= ceiling * 1.08), "review", "reject"),
    )

    df = pd.DataFrame(
        {
            "deal_id": [f"DHR-{i:06d}" for i in range(n_rows)],
            "service_line": service,
            "consultant_seniority": seniority,
            "region": region,
            "client_size": client_size,
            "contract_months": contract_months,
            "skill_score": np.round(skills, 2),
            "urgency_score": urgency,
            "utilization_rate": np.round(utilization, 3),
            "market_rate": np.round(market_rate, 2),
            "cost_rate": np.round(cost_rate, 2),
            "proposed_rate": np.round(proposed_rate, 2),
            "discount_pct": np.round(discount_pct, 4),
            "proposed_margin": np.round(proposed_margin, 4),
            "win_probability": np.round(win_probability, 4),
            "pricing_decision": status,
        }
    )

    messy_idx = rng.choice(df.index, size=int(n_rows * 0.04), replace=False)
    df.loc[messy_idx[: len(messy_idx) // 3], "proposed_rate"] = np.nan
    df.loc[messy_idx[len(messy_idx) // 3 : 2 * len(messy_idx) // 3], "region"] = " west "
    df.loc[messy_idx[2 * len(messy_idx) // 3 :], "skill_score"] = 999
    dupes = df.sample(80, random_state=seed)
    return pd.concat([df, dupes], ignore_index=True)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = create_synthetic_hr_pricing_data()
    data.to_csv(RAW_DIR / "damohr_pricing_raw.csv", index=False)
    print(f"Created {RAW_DIR / 'damohr_pricing_raw.csv'} with {len(data):,} rows")

