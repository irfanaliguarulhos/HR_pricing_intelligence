from pathlib import Path

import pandas as pd

from damohr.data_preparation import clean_with_pyspark
from damohr.eda_stats import run_econometric_checks, run_statistical_tests
from damohr.generate_data import create_synthetic_hr_pricing_data
from damohr.modeling import train_models


ROOT = Path(__file__).resolve().parents[2]


def main():
    raw = ROOT / "data" / "raw" / "damohr_pricing_raw.csv"
    clean = ROOT / "data" / "processed" / "damohr_pricing_clean.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    clean.parent.mkdir(parents=True, exist_ok=True)
    create_synthetic_hr_pricing_data().to_csv(raw, index=False)
    df = clean_with_pyspark(raw, clean)
    stats = run_statistical_tests(df)
    econ = run_econometric_checks(df)
    model_summary = train_models(pd.read_csv(clean))
    print({"stats": stats, "econometrics": econ, "final_model": model_summary["final_model"]})


if __name__ == "__main__":
    main()

