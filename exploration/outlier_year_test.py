"""
SP5 — Outlier Year Test
Tests whether the D1 main effect sign instability is driven by a single outlier year.

Steps:
1. Statistical outlier detection on D1 annual values (IQR and z-score)
2. Leave-one-out: drop each year and measure interaction coefficient stability
3. Key question: are interaction coefficients stable, or driven by one year?
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
TUS_PATH = Path("data/processed/tus_panel_macro.csv")
D1_PATH  = Path("data/processed/gnews_d1_sentiment_final.csv")
D2_PATH  = Path("data/processed/eksi_d2_yearly.csv")
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Load + prep (mevcut kodunla aynı)
# ---------------------------------------------------------------------------

tus = pd.read_csv(TUS_PATH).dropna(subset=["score_min"])
d1  = pd.read_csv(D1_PATH)[["year", "d1_count"]]
d2  = pd.read_csv(D2_PATH)[["year", "d2_violence_proportion"]].rename(
    columns={"d2_violence_proportion": "d2_annot"})

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

tus = tus[~tus["branch"].isin(EXCLUDE)]
tus["exposure"] = tus["branch"].apply(
    lambda b: "high" if b in HIGH_EXPOSURE else ("low" if b in LOW_EXPOSURE else "intermediate")
)
tus["post_reform"] = (tus["year"] >= 2022).astype(int)
tus["high_exp"]    = (tus["exposure"] == "high").astype(int)
tus["mid_exp"]     = (tus["exposure"] == "intermediate").astype(int)

tus = tus.drop(columns=[c for c in tus.columns if c in [
    "d1_count", "d1_binary", "d1_keyword_coverage",
    "n_violence", "violence_ratio", "d1_count_z", "d2_rate_z", "d2_rate",
    "cpi_annual_mean", "usd_try"
]], errors="ignore")

for df in [d1, d2, tus]:
    df["year"] = df["year"].astype(int)

panel = tus.merge(d1, on="year", how="left").merge(d2, on="year", how="left")

# ---------------------------------------------------------------------------
# PART 1 — D1 outlier testi (year-level, N=13)
# ---------------------------------------------------------------------------

annual_d1 = panel.drop_duplicates("year")[["year", "d1_count"]].sort_values("year")

mean_d1 = annual_d1["d1_count"].mean()
std_d1  = annual_d1["d1_count"].std()
q1      = annual_d1["d1_count"].quantile(0.25)
q3      = annual_d1["d1_count"].quantile(0.75)
iqr     = q3 - q1

annual_d1["z_score"]      = (annual_d1["d1_count"] - mean_d1) / std_d1
annual_d1["iqr_outlier"]  = (
    (annual_d1["d1_count"] < q1 - 1.5 * iqr) |
    (annual_d1["d1_count"] > q3 + 1.5 * iqr)
)
annual_d1["z_outlier"]    = annual_d1["z_score"].abs() > 2.0

SEP = "=" * 65

print(SEP)
print("PART 1 — D1 Annual Values + Outlier Flags")
print(f"Mean={mean_d1:.1f}  SD={std_d1:.1f}  IQR={iqr:.1f}")
print(f"IQR fence: [{q1 - 1.5*iqr:.1f}, {q3 + 1.5*iqr:.1f}]")
print(f"Z-score threshold: ±2.0")
print(SEP)
print(f"\n{'Year':>6}  {'D1':>8}  {'Z-score':>8}  {'IQR_outlier':>12}  {'Z_outlier':>10}")
print("-" * 55)
for _, row in annual_d1.iterrows():
    flag = " ← OUTLIER" if row["iqr_outlier"] or row["z_outlier"] else ""
    print(f"  {int(row['year']):4d}  {row['d1_count']:8.1f}  {row['z_score']:8.2f}  "
          f"{'YES' if row['iqr_outlier'] else 'no':>12}  "
          f"{'YES' if row['z_outlier'] else 'no':>10}  {flag}")

outlier_years = annual_d1[annual_d1["iqr_outlier"] | annual_d1["z_outlier"]]["year"].tolist()
print(f"\nFlagged years: {outlier_years}")

# ---------------------------------------------------------------------------
# PART 2 — Leave-one-out: her yılı çıkarıp interaction stabilitesini ölç
# ---------------------------------------------------------------------------

print(f"\n{SEP}")
print("PART 2 — Leave-One-Out: Interaction Coefficient Stability")
print("Primary spec: D1 only | quadratic trend + post_reform")
print("Key question: do interaction coefficients change when we drop a year?")
print(SEP)

def build_agg(data):
    """Build year-level aggregated dataset from panel."""
    for col in ["d1_count", "d2_annot"]:
        if col in data.columns:
            data[f"{col}_z"] = (data[col] - data[col].mean()) / data[col].std()
    data["year_c"]  = data["year"] - data["year"].mean()
    data["year_c2"] = data["year_c"] ** 2
    data["d1_count_z_x_high"] = data["d1_count_z"] * data["high_exp"]
    data["d1_count_z_x_mid"]  = data["d1_count_z"] * data["mid_exp"]

    agg = data.groupby(["year", "exposure"]).agg(
        score_min         = ("score_min",            "mean"),
        d1_count_z        = ("d1_count_z",            "first"),
        year_c            = ("year_c",                "first"),
        year_c2           = ("year_c2",               "first"),
        post_reform       = ("post_reform",            "first"),
        high_exp          = ("high_exp",               "first"),
        mid_exp           = ("mid_exp",                "first"),
        d1_count_z_x_high = ("d1_count_z_x_high",     "first"),
        d1_count_z_x_mid  = ("d1_count_z_x_mid",      "first"),
    ).reset_index()
    return agg

FORMULA = ("score_min ~ d1_count_z + d1_count_z_x_high + d1_count_z_x_mid "
           "+ high_exp + mid_exp + year_c + year_c2 + post_reform")

KEY_VARS = ["d1_count_z", "d1_count_z_x_high", "d1_count_z_x_mid"]

print(f"\n{'Dropped':>8}  {'N':>4}  ", end="")
print("  ".join(f"{v.replace('d1_count_z','D1').replace('_x_','×'):>18}" for v in KEY_VARS))
print("-" * 80)

loo_results = []

# Full sample baseline
agg_full = build_agg(panel.copy())
m_full   = smf.ols(FORMULA, data=agg_full).fit()
row_data  = {"dropped_year": "FULL SAMPLE"}
print(f"{'BASELINE':>8}  {int(m_full.nobs):>4}  ", end="")
for var in KEY_VARS:
    c = m_full.params.get(var, np.nan)
    p = m_full.pvalues.get(var, np.nan)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {c:+.3f}{sig:3s}(p={p:.3f})    ", end="")
    row_data[f"{var}_coef"] = round(c, 3)
    row_data[f"{var}_p"]    = round(p, 3)
print()
loo_results.append(row_data)

# Leave-one-out
years = sorted(panel["year"].unique())
for yr in years:
    panel_loo = panel[panel["year"] != yr].copy()
    try:
        agg_loo = build_agg(panel_loo)
        m_loo   = smf.ols(FORMULA, data=agg_loo).fit()
        row_data = {"dropped_year": yr}
        flag = " ← FLAGGED" if yr in outlier_years else ""
        print(f"  {yr:>6}{flag[:2]:>2}  {int(m_loo.nobs):>4}  ", end="")
        for var in KEY_VARS:
            c = m_loo.params.get(var, np.nan)
            p = m_loo.pvalues.get(var, np.nan)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {c:+.3f}{sig:3s}(p={p:.3f})    ", end="")
            row_data[f"{var}_coef"] = round(c, 3)
            row_data[f"{var}_p"]    = round(p, 3)
        print(flag)
        loo_results.append(row_data)
    except Exception as e:
        print(f"  {yr:>6}  FAILED: {e}")

# ---------------------------------------------------------------------------
# PART 3 — Explicitly drop flagged outlier years and re-run
# ---------------------------------------------------------------------------

print(f"\n{SEP}")
print("PART 3 — Explicit Outlier Exclusion")
print(f"Dropping flagged years: {outlier_years}")
print("Compares: full sample vs outlier-excluded")
print(SEP)

if outlier_years:
    panel_excl = panel[~panel["year"].isin(outlier_years)].copy()
    agg_excl   = build_agg(panel_excl)
    m_excl     = smf.ols(FORMULA, data=agg_excl).fit()

    print(f"\n{'Variable':32s}  {'Full sample':>14}  {'Excl. outlier':>14}  {'Stable?':>8}")
    print("-" * 75)
    for var in KEY_VARS:
        c_full  = m_full.params.get(var, np.nan)
        p_full  = m_full.pvalues.get(var, np.nan)
        c_excl  = m_excl.params.get(var, np.nan)
        p_excl  = m_excl.pvalues.get(var, np.nan)
        sig_f   = "***" if p_full < 0.01 else "**" if p_full < 0.05 else "*" if p_full < 0.1 else ""
        sig_e   = "***" if p_excl < 0.01 else "**" if p_excl < 0.05 else "*" if p_excl < 0.1 else ""
        # Sign stable + both significant = stable
        sign_stable = np.sign(c_full) == np.sign(c_excl)
        sig_stable  = (p_full < 0.1) == (p_excl < 0.1)
        stable = "YES" if (sign_stable and sig_stable) else "NO ←"
        print(f"  {var:30s}  {c_full:+6.3f}{sig_f:3s}          {c_excl:+6.3f}{sig_e:3s}          {stable}")

    print(f"\n  Full sample N={int(m_full.nobs)}, R²={m_full.rsquared:.3f}")
    print(f"  Excl. sample N={int(m_excl.nobs)}, R²={m_excl.rsquared:.3f}")
else:
    print("\nNo years flagged as outliers — no exclusion needed.")
    print("D1 main effect instability is likely a trend specification issue, not outlier-driven.")

# ---------------------------------------------------------------------------
# PART 4 — Interpretation guide
# ---------------------------------------------------------------------------

print(f"\n{SEP}")
print("INTERPRETATION GUIDE")
print(SEP)
print("""
If interaction coefficients are STABLE across leave-one-out:
  → The differential effect is robust; no single year is driving the result.
  → D1 main effect instability is a trend-control artifact (as already documented).
  → Conclusion: report interactions as primary finding, note main effect instability.

If interaction coefficients CHANGE SIGN or lose significance when dropping a year:
  → That year is influential. Investigate why:
      (a) Unrelated political/economic shock in that year?
      (b) Data collection anomaly (RSS truncation, coverage change)?
  → If (a) or (b) is defensible on PRIOR grounds, exclude and report as sensitivity.
  → If no prior justification, do NOT exclude — report as a limitation instead.

2022 note: 2022 is theoretically the most important year (Ekrem Karakaya shooting,
D1 peak). Excluding it requires a strong non-circular justification.
""")
