from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def clean_with_pandas(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.drop_duplicates(subset=["deal_id"])
    for col in ["service_line", "consultant_seniority", "region", "client_size"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df["region"] = df["region"].replace({"West": "West"})
    df["skill_score"] = df["skill_score"].clip(0, 100)
    numeric_cols = ["proposed_rate", "market_rate", "cost_rate", "proposed_margin", "win_probability"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    df = df[(df["proposed_rate"] > 0) & (df["market_rate"] > 0) & (df["cost_rate"] > 0)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def clean_with_pyspark(input_path: Path, output_path: Path) -> pd.DataFrame:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except Exception:
        return clean_with_pandas(input_path, output_path)

    spark = SparkSession.builder.appName("DamoHRPricingPreparation").getOrCreate()
    sdf = spark.read.csv(str(input_path), header=True, inferSchema=True)
    sdf = sdf.dropDuplicates(["deal_id"])
    for col in ["service_line", "consultant_seniority", "region", "client_size"]:
        sdf = sdf.withColumn(col, F.initcap(F.trim(F.col(col))))
    median_rate = sdf.approxQuantile("proposed_rate", [0.5], 0.01)[0]
    sdf = sdf.fillna({"proposed_rate": median_rate})
    sdf = sdf.withColumn("skill_score", F.when(F.col("skill_score") > 100, 100).otherwise(F.col("skill_score")))
    sdf = sdf.filter((F.col("proposed_rate") > 0) & (F.col("market_rate") > 0) & (F.col("cost_rate") > 0))
    df = sdf.toPandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    spark.stop()
    return df


if __name__ == "__main__":
    clean_with_pyspark(ROOT / "data" / "raw" / "damohr_pricing_raw.csv", ROOT / "data" / "processed" / "damohr_pricing_clean.csv")

