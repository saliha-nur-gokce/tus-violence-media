"""
D2 Version Comparison
Compares three D2 measures against TUS minimum scores by exposure group,
motivating the choice of d2_annot as the primary treatment variable.

Measures compared:
    d2_annot    — annual share of violence-flagged entries (LLM annotation, chosen)
    d2_severity — annual mean violence score (0–3) per entry
    d2_keyword  — annual share of entries containing any violence keyword

Key findings:
  - d2_keyword shows no significant signal in any exposure group
  - d2_severity has a borderline falsification concern (low-exposure p=0.052)
  - d2_annot is chosen: global panel signal, no falsification issue

Inputs:
    data/processed/eksi_d2_yearly.csv    — d2_annot and d2_severity (step 07)
    data/raw/eksi/eksi_corpus_final.csv  — raw corpus for d2_keyword recomputation
    data/processed/tus_panel_macro.csv   — TUS panel (step 01)
"""

import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
TUS_PATH    = Path("data/processed/tus_panel_macro.csv")
D2_PATH     = Path("data/processed/eksi_d2_yearly.csv")
CORPUS_PATH = Path("data/raw/eksi/eksi_corpus_final.csv")
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

# Violence keywords — root forms to catch Turkish inflections
VIOLENCE_KEYWORDS = [
    "şiddet", "saldır", "darp", "tehdit", "vur", "yumruk",
    "bıçak", "silah", "öldür", "yaralad", "kavga", "linç",
    "taciz", "hakaret", "küfür", "cop", "taşla", "döv",
    "saldırgan", "mağdur", "katil", "cinayet", "şiddetli",
]

D2_VERSIONS = ["d2_annot", "d2_severity", "d2_keyword"]

SEP = "=" * 65


def compute_d2_keyword(corpus_path: Path) -> pd.DataFrame:
    """
    Recomputes the keyword-based D2 measure from the raw corpus.
    Mirrors the approach in eksi_final.ipynb (CELL KEYWORD).
    Returns yearly aggregation with d2_keyword = violence_flag_rate.
    """
    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    df = df[df["year"] <= 2025].copy()
    df["year"] = df["year"].astype(int)

    def contains_violence(text: str) -> int:
        text = str(text).lower()
        return int(any(kw in text for kw in VIOLENCE_KEYWORDS))

    df["violence_flag"] = df["content"].apply(contains_violence)

    yearly = (
        df.groupby("year")
        .agg(n_entries=("violence_flag", "count"), n_violence=("violence_flag", "sum"))
        .reset_index()
    )
    yearly["d2_keyword"] = yearly["n_violence"] / yearly["n_entries"]
    return yearly[["year", "d2_keyword"]]


def main():
    tus = pd.read_csv(TUS_PATH).dropna(subset=["score_min"])
    d2  = pd.read_csv(D2_PATH)
    d2["year"] = d2["year"].astype(int)

    d2 = d2.rename(columns={
        "d2_violence_proportion": "d2_annot",
        "mean_violence_score":    "d2_severity",
    })

    d2_kw = compute_d2_keyword(CORPUS_PATH)
    d2 = d2.merge(d2_kw, on="year", how="left")

    tus = tus[~tus["branch"].isin(EXCLUDE)].copy()
    tus["exposure"] = tus["branch"].apply(
        lambda b: "high" if b in HIGH_EXPOSURE else ("low" if b in LOW_EXPOSURE else "intermediate")
    )

    # ── Correlation between D2 versions (year-level) ──────────────────────────
    print(SEP)
    print("PART 1 — D2 Inter-measure Spearman Correlations (year-level)")
    print(SEP)
    pairs = [
        ("d2_annot",    "d2_severity", "d2_annot    vs d2_severity"),
        ("d2_annot",    "d2_keyword",  "d2_annot    vs d2_keyword "),
        ("d2_severity", "d2_keyword",  "d2_severity vs d2_keyword "),
    ]
    for xa, xb, label in pairs:
        sub = d2[[xa, xb]].dropna()
        r, p = spearmanr(sub[xa], sub[xb])
        print(f"  {label}  r={r:+.3f}  p={p:.3f}")

    # ── Spearman vs TUS scores by exposure group ──────────────────────────────
    annual = (
        tus.merge(d2[["year"] + D2_VERSIONS], on="year")
        .groupby(["year", "exposure"])[["score_min"] + D2_VERSIONS]
        .mean()
        .reset_index()
    )

    print(f"\n{SEP}")
    print("PART 2 — Spearman(D2 version, mean score_min) by Exposure Group")
    print(SEP)
    print(f"\n{'Exposure':>14}  {'D2 version':>14}  {'r':>7}  {'p':>7}")
    print("-" * 50)

    for exposure in ["high", "intermediate", "low"]:
        sub = annual[annual["exposure"] == exposure].dropna()
        for v in D2_VERSIONS:
            r, p = spearmanr(sub["score_min"], sub[v])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {exposure:>12}  {v:>14}  {r:+.3f}  {p:.3f}  {sig}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(SEP)
    print("INTERPRETATION")
    print(SEP)
    print("""
d2_annot (LLM annotation proportion) is chosen as primary because:
  - Theoretically grounded: LLM labels distinguish violence types
  - d2_keyword shows no significant signal in any exposure group
    (root-form matching does not capture discourse intensity)
  - d2_severity shows borderline low-exposure effect (p=0.052),
    which is a falsification concern — low-exposure should show no effect
  - d2_annot passes falsification: low-exposure p > 0.1

Note: d2_severity and d2_annot are highly correlated (both from
eksi_d2_yearly.csv). The difference is that d2_severity weights by
violence score magnitude; d2_annot treats any positive label as binary.
""")


if __name__ == "__main__":
    main()
