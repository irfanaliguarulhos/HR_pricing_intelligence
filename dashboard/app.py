from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "damohr_pricing_clean.csv"
KB_PATH = ROOT / "rag" / "knowledge_base.md"

st.set_page_config(page_title="DamoHR Pricing Intelligence", layout="wide")
st.title("DamoHR Pricing Intelligence")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deals", f"{len(df):,}")
    c2.metric("Approval Rate", f"{(df.pricing_decision.eq('approve').mean() * 100):.1f}%")
    c3.metric("Median Margin", f"{df.proposed_margin.median() * 100:.1f}%")
    c4.metric("Median Rate", f"${df.proposed_rate.median():,.0f}")
    st.bar_chart(df["pricing_decision"].value_counts())
    st.scatter_chart(df, x="market_rate", y="proposed_rate", color="pricing_decision")
else:
    st.warning("Run `python src/damohr/run_pipeline.py` to create processed data.")

st.subheader("RAG Interview Assistant")
question = st.text_input("Ask about pricing, validation, RAG, MLOps, or econometrics")
if question and KB_PATH.exists():
    kb = KB_PATH.read_text(encoding="utf-8")
    hits = [line for line in kb.splitlines() if any(token.lower() in line.lower() for token in question.split())]
    st.write("Retrieved context:")
    st.write("\n".join(hits[:8]) or "No exact match found. Use the knowledge base as fallback context.")

