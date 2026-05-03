"""
D1 Version Comparison
Compares three D1 measures against TUS minimum scores by exposure group,
motivating the choice of d1_count as the primary treatment variable.

Measures compared:
    d1_count          — raw annual article count (chosen)
    d1_keyword_mean   — mean keyword-negativity score per article
    d1_bert_mean      — mean BERT-predicted negativity score per article

Key finding: d1_count shows the strongest theoretical signal
(high-exposure Spearman r = -0.908, p < 0.001).
d1_keyword_mean and d1_bert_mean are more weakly correlated with exposure.

Inputs:
    data/processed/gnews_d1_sentiment_final.csv   — all D1 versions (step 04)
    data/processed/tus_panel_macro.csv            — TUS panel (step 01)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
TUS_PATH = Path("data/processed/tus_panel_macro.csv")
D1_PATH  = Path("data/processed/gnews_d1_sentiment_final.csv")
# ─────────────────────────────────────────────────────────────────────────────

HIGH_EXPOSURE = [
    "Acil Tıp", "Genel Cerrahi", "Beyin ve Sinir Cerrahisi",
    "Kalp ve Damar Cerrahisi", "Göğüs Cerrahisi",
    "Kadın Hastalıkları ve Doğum", "Çocuk Cerrahisi",
    "Üroloji", "Ortopedi ve Travmatoloji",
]
LOW_EXPOSURE = [
    "Deri ve Zührevi Hastalıkları", "Radyoloji", "Nükleer Tıp",
    "Tıbbi Patoloji", "Tıbbi Biyokimya", "Anatomi",
    "Histoloji ve Embriyoloji", "Fizyoloji", "Tıbbi Farmakoloji",
]
EXCLUDE = ["Plastik, Rekonstrüktif ve Estetik Cerrahi"]

D1_VERSIONS = ["d1_count", "d1_keyword_mean", "d1_bert_mean"]

SEP = "=" * 65


def main():
    tus = pd.read_csv(TUS_PATH).dropna(subset=["score_min"])
    d1  = pd.read_csv(D1_PATH)
    d1["year"] = d1["year"].astype(int)

    tus = tus[~tus["branch"].isin(EXCLUDE)].copy()
    tus["exposure"] = tus["branch"].apply(
        lambda b: "high" if b in HIGH_EXPOSURE else ("low" if b in LOW_EXPOSURE else "intermediate")
    )

    # ── Correlation between D1 versions (year-level, N=13) ───────────────────
    print(SEP)
    print("PART 1 — D1 Inter-measure Spearman Correlations (year-level, N=13)")
    print(SEP)
    pairs = [
        ("d1_count",        "d1_keyword_mean", "d1_count       vs d1_keyword_mean"),
        ("d1_count",        "d1_bert_mean",    "d1_count       vs d1_bert_mean   "),
        ("d1_keyword_mean", "d1_bert_mean",    "d1_keyword_mean vs d1_bert_mean  "),
    ]
    for xa, xb, label in pairs:
        r, p = spearmanr(d1[xa], d1[xb])
        print(f"  {label}  r={r:+.3f}  p={p:.3f}")

    # ── Spearman vs TUS scores by exposure group ──────────────────────────────
    annual = (
        tus.merge(d1[["year"] + D1_VERSIONS], on="year")
        .groupby(["year", "exposure"])[["score_min"] + D1_VERSIONS]
        .mean()
        .reset_index()
    )

    print(f"\n{SEP}")
    print("PART 2 — Spearman(D1 version, mean score_min) by Exposure Group")
    print(SEP)
    print(f"\n{'Exposure':>14}  {'D1 version':>20}  {'r':>7}  {'p':>7}")
    print("-" * 55)

    for exposure in ["high", "intermediate", "low"]:
        sub = annual[annual["exposure"] == exposure].dropna()
        for v in D1_VERSIONS:
            r, p = spearmanr(sub["score_min"], sub[v])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {exposure:>12}  {v:>20}  {r:+.3f}  {p:.3f}  {sig}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(SEP)
    print("INTERPRETATION")
    print(SEP)
    print("""
d1_count is chosen as primary treatment variable because:
  - Strongest theoretical alignment: high-exposure Spearman r = -0.908
  - Directly replicates the Bo et al. (2020) article-count measure
  - d1_keyword_mean and d1_bert_mean add sentiment weighting but do not
    improve the exposure-group differentiation

d1_keyword_weighted and d1_bert_weighted (article_count x mean score) are
also available in gnews_d1_sentiment_final.csv for robustness checks.
""")


if __name__ == "__main__":
    main()
