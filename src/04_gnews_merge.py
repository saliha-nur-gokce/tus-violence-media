"""
Pipeline step 04 — Google News Title Cleaning, Keyword Scoring & D1 Merge
Inputs:
    gnews_articles_raw.csv          — raw article-level data from step 02
    gnews_yearly_bert_sentiment.csv — yearly BERT scores from step 03 (Colab)
Outputs:
    gnews_scored.csv                — cleaned titles + keyword scores (also feeds step 03)
    gnews_yearly_keyword_sentiment.csv — yearly keyword sentiment aggregate
    gnews_d1_sentiment_final.csv    — final D1 file for regression (step 08)

Run order:
    1. Run prepare()  → produces gnews_scored.csv
    2. Run step 03 on Colab using gnews_scored.csv
    3. Download gnews_yearly_bert_sentiment.csv from Drive
    4. Run merge()    → produces gnews_d1_sentiment_final.csv

    Calling main() handles both phases automatically; it skips the merge
    if the BERT file is not yet available.

Primary D1 variable: d1_count (plain article count, chosen over keyword/BERT
weighted variants — see CONTEXT.md for selection rationale).
"""

import re
import sys
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
INPUT_RAW         = Path("data/raw/gnews/gnews_articles_raw.csv")
OUTPUT_SCORED     = Path("data/raw/gnews/gnews_scored.csv")
OUTPUT_KW_YEARLY  = Path("data/processed/gnews_yearly_keyword_sentiment.csv")
INPUT_BERT_YEARLY = Path("data/processed/gnews_yearly_bert_sentiment.csv")
OUTPUT_FINAL      = Path("data/processed/gnews_d1_sentiment_final.csv")
# ─────────────────────────────────────────────────────────────────────────────


# ── TITLE CLEANING ────────────────────────────────────────────────────────────

def _turkish_lower(text: str) -> str:
    """
    Turkish-safe lowercase.
    Python's str.lower() converts İ (U+0130) to i + combining dot above (U+0307),
    which breaks substring matching. Replace İ->i and I->ı before calling lower().
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


def _strip_source_suffix(text: str) -> str:
    """Remove trailing source name appended to titles (e.g. ' - Sabah', ' | Hürriyet')."""
    return re.sub(r"\s*[\|\-–]\s*[^|\-–]{2,60}$", "", text).strip()


def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_clean"] = (
        df["title"]
        .astype(str)
        .str.strip()
        .apply(_strip_source_suffix)
        .apply(_turkish_lower)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # Combining dot (U+0307) must not appear — signals a broken İ conversion
    bad = df["title_clean"].str.contains("̇").sum()
    assert bad == 0, f"{bad} titles contain combining dot (U+0307) after cleaning"
    return df


# ── KEYWORD SCORING ───────────────────────────────────────────────────────────

NEGATIVE_KEYWORDS = [
    "saldırı", "saldırıya", "saldırdı", "saldıran",
    "darp", "darbedildi", "darbetti",
    "dövüldü", "dövdü", "dayak",
    "bıçaklandı", "bıçakla",
    "vuruldu", "vurdu", "yaralanan",
    "öldürüldü", "öldürülen", "ölen",
    "yaralandı", "yaralı", "yaraladı",
    "şiddet", "şiddete", "şiddetle",
    "tehdit", "tehdit edildi",
    "hakaret", "küfür",
    "linç", "taciz",
    "dehşet", "katliam", "vahşet", "ölü",
    "ölüm", "öldü", "hayatını kaybetti",
    "kınıyoruz", "kınadı",
    "protesto", "suç duyurusu",
]

POSITIVE_KEYWORDS = [
    "önlem", "koruma", "destek", "iyileşti", "taburcu", "başarılı",
]


def _keyword_neg_score(text: str) -> float:
    """
    Normalised negativity score in [0, 1].
    Formula: max(0, neg_hits - pos_hits) / sqrt(word_count).
    sqrt normalisation reduces length bias without over-penalising short titles.
    """
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    wc  = max(len(text.split()), 1)
    return min(1.0, max(0, neg - pos) / (wc ** 0.5))


def _keyword_neg_flag(text: str) -> int:
    return int(any(kw in text for kw in NEGATIVE_KEYWORDS))


def add_keyword_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["keyword_neg_score"] = df["title_clean"].apply(_keyword_neg_score)
    df["keyword_neg_flag"]  = df["title_clean"].apply(_keyword_neg_flag)
    return df


# ── YEARLY AGGREGATION ────────────────────────────────────────────────────────

def build_keyword_yearly(df: pd.DataFrame) -> pd.DataFrame:
    treatment = df[df["group"] == "treatment"]
    yearly = (
        treatment.groupby("year")
        .agg(
            article_count  = ("title_clean",       "count"),
            mean_neg_score = ("keyword_neg_score", "mean"),
            neg_flag_rate  = ("keyword_neg_flag",  "mean"),
            neg_flag_count = ("keyword_neg_flag",  "sum"),
        )
        .reset_index()
    )
    yearly["d1_keyword_weighted"] = yearly["article_count"] * yearly["mean_neg_score"]
    return yearly


# ── PHASE 1: PREPARE ──────────────────────────────────────────────────────────

def prepare() -> None:
    """
    Phase 1: clean titles, add keyword scores, save gnews_scored.csv.
    Run this before step 03 (Colab BERT).
    """
    print("=== Phase 1: title cleaning + keyword scoring ===")

    df = pd.read_csv(INPUT_RAW)
    print(f"Loaded: {df.shape[0]} rows")

    # Drop any incomplete/future year that slipped in
    df = df[df["year"] <= 2025].copy()

    df = clean_titles(df)
    df = add_keyword_scores(df)

    OUTPUT_SCORED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_SCORED, index=False, encoding="utf-8-sig")
    print(f"Saved scored articles : {OUTPUT_SCORED}")

    for grp in ["treatment", "control"]:
        sub  = df[df["group"] == grp]
        rate = sub["keyword_neg_flag"].mean()
        print(f"  Neg flag rate ({grp:10s}): {rate:.3f}  (n={len(sub)})")

    yearly_kw = build_keyword_yearly(df)
    OUTPUT_KW_YEARLY.parent.mkdir(parents=True, exist_ok=True)
    yearly_kw.to_csv(OUTPUT_KW_YEARLY, index=False, encoding="utf-8-sig")
    print(f"Saved keyword yearly  : {OUTPUT_KW_YEARLY}")

    print(f"\nPhase 1 done. Upload {OUTPUT_SCORED} to Google Drive, then run step 03.")


# ── PHASE 2: MERGE ────────────────────────────────────────────────────────────

def merge() -> None:
    """
    Phase 2: merge keyword + BERT yearly scores → gnews_d1_sentiment_final.csv.
    Run after step 03 (Colab) completes and BERT output is downloaded locally.
    """
    print("=== Phase 2: merging keyword and BERT yearly scores ===")

    if not INPUT_BERT_YEARLY.exists():
        print(f"ERROR: {INPUT_BERT_YEARLY} not found.")
        print("Run step 03 on Colab first, then download gnews_yearly_bert_sentiment.csv.")
        sys.exit(1)

    kw   = pd.read_csv(OUTPUT_KW_YEARLY)
    bert = pd.read_csv(INPUT_BERT_YEARLY)

    merged = kw.merge(
        bert[["year", "bert_mean_neg", "bert_neg_flag_rate", "d1_bert_weighted"]],
        on="year",
    )

    print("\n=== Merged yearly scores ===")
    print(merged[["year", "article_count", "mean_neg_score",
                  "bert_mean_neg", "d1_keyword_weighted",
                  "d1_bert_weighted"]].to_string(index=False))

    sr, sp = spearmanr(merged["mean_neg_score"], merged["bert_mean_neg"])
    pr, pp = pearsonr(merged["mean_neg_score"],  merged["bert_mean_neg"])
    print(f"\nKeyword mean vs BERT mean: Spearman r={sr:.3f} p={sp:.3f} | Pearson r={pr:.3f} p={pp:.3f}")

    sr2, sp2 = spearmanr(merged["d1_keyword_weighted"], merged["d1_bert_weighted"])
    pr2, pp2 = pearsonr(merged["d1_keyword_weighted"],  merged["d1_bert_weighted"])
    print(f"Keyword wtd  vs BERT wtd : Spearman r={sr2:.3f} p={sp2:.3f} | Pearson r={pr2:.3f} p={pp2:.3f}")

    final = merged[[
        "year", "article_count",
        "d1_keyword_weighted", "d1_bert_weighted",
        "mean_neg_score", "bert_mean_neg",
    ]].copy()
    final.columns = [
        "year", "d1_count",
        "d1_keyword_weighted", "d1_bert_weighted",
        "d1_keyword_mean", "d1_bert_mean",
    ]

    OUTPUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"\nSaved final D1 file : {OUTPUT_FINAL}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    prepare()
    if INPUT_BERT_YEARLY.exists():
        merge()
    else:
        print(f"\n[SKIP] {INPUT_BERT_YEARLY} not found — skipping merge phase.")
        print("After step 03 completes, re-run this script to produce the final D1 file.")


if __name__ == "__main__":
    main()
