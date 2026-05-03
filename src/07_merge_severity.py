"""
Pipeline step 07 — Eksi Sozluk Annotation Merge + D2 Construction
Inputs:
    data/raw/eksi/eksi_corpus_final.csv              — full corpus from step 05
    data/processed/eksi_annotation_v2_labeled_a.csv  — human annotations (annotator A)
    data/processed/eksi_annotation_v2_labeled_b.csv  — human annotations (annotator B)
    data/processed/eksi_llm_annotated.csv            — LLM annotations from step 06
Outputs:
    data/processed/eksi_annotated_full.csv           — full corpus with labels
    data/processed/eksi_d2_yearly.csv                — yearly D2 aggregates (regression input)

Human label resolution:
    - For entries where both annotators agree: use agreed label
    - For disagreements: take the average and round (conservative)
    - Annotator B applied the more conservative standard; used as ground truth
      for few-shot examples in step 06

D2 variable:
    Primary: d2_violence_proportion (share of violent entries per year)
    Also computed: d2_weighted (entry_count * mean_violence_score)
"""

import pandas as pd
from pathlib import Path

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
INPUT_CORPUS  = Path("data/raw/eksi/eksi_corpus_final.csv")
INPUT_ANNOT_A = Path("data/processed/eksi_annotation_v2_labeled_a.csv")
INPUT_ANNOT_B = Path("data/processed/eksi_annotation_v2_labeled_b.csv")
INPUT_LLM     = Path("data/processed/eksi_llm_annotated.csv")
OUTPUT_FULL   = Path("data/processed/eksi_annotated_full.csv")
OUTPUT_D2     = Path("data/processed/eksi_d2_yearly.csv")
# ─────────────────────────────────────────────────────────────────────────────

COLS = [
    "physical_violence",
    "verbal_violence",
    "violence_normalized",
    "anti_violence",
    "sarcasm",
]


def merge_human_labels(
    annot_a: pd.DataFrame,
    annot_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges the two human annotators' labels into a single set.
    For agreed entries the shared label is used directly.
    For disagreements the average is rounded (conservative tiebreak).
    Returns a DataFrame with entry_id, COLS, and annotator='human'.
    """
    merged = annot_a[["entry_id"] + COLS].merge(
        annot_b[["entry_id"] + COLS],
        on="entry_id",
        suffixes=("_a", "_b"),
    ).dropna()

    for col in COLS:
        merged[f"{col}_a"] = merged[f"{col}_a"].astype(int)
        merged[f"{col}_b"] = merged[f"{col}_b"].astype(int)

    human = pd.DataFrame({"entry_id": merged["entry_id"]})
    for col in COLS:
        human[col] = ((merged[f"{col}_a"] + merged[f"{col}_b"]) / 2).round().astype(int)

    human["annotator"] = "human"
    return human


def build_full_corpus(
    corpus: pd.DataFrame,
    human_labels: pd.DataFrame,
    llm_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines human and LLM labels, merges with corpus metadata,
    and computes entry-level violence score and flag.
    """
    llm = llm_labels[["entry_id"] + COLS].copy()
    for col in COLS:
        llm[col] = llm[col].astype(int)
    llm["annotator"] = "llm"

    all_labels = pd.concat([human_labels, llm], ignore_index=True)
    print(f"Total labeled entries : {len(all_labels)}")

    full = corpus.merge(all_labels, on="entry_id", how="left")
    missing = full[COLS[0]].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} corpus entries have no labels")

    # violence_score: sum of physical + verbal + normalized (range 0–3)
    full["violence_score"] = (
        full["physical_violence"].fillna(0).astype(int)
        + full["verbal_violence"].fillna(0).astype(int)
        + full["violence_normalized"].fillna(0).astype(int)
    )
    full["violence_flag"] = (full["violence_score"] > 0).astype(int)

    return full


def build_d2_yearly(full: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates entry-level labels to yearly D2 variables.

    d2_violence_proportion : share of entries flagged as violent (primary D2)
    d2_weighted            : entry_count * mean_violence_score
    """
    yearly = (
        full.groupby("year")
        .agg(
            entry_count         = ("entry_id",        "count"),
            mean_violence_score = ("violence_score",  "mean"),
            violence_flag_rate  = ("violence_flag",   "mean"),
            violence_flag_count = ("violence_flag",   "sum"),
            anti_violence_rate  = ("anti_violence",   "mean"),
            sarcasm_rate        = ("sarcasm",         "mean"),
        )
        .reset_index()
    )

    yearly["d2_violence_proportion"] = yearly["violence_flag_rate"]
    yearly["d2_weighted"]            = yearly["entry_count"] * yearly["mean_violence_score"]

    return yearly


def main() -> None:
    corpus  = pd.read_csv(INPUT_CORPUS,  encoding="utf-8-sig")
    annot_a = pd.read_csv(INPUT_ANNOT_A, sep=None, engine="python")
    annot_b = pd.read_csv(INPUT_ANNOT_B, sep=None, engine="python")
    llm     = pd.read_csv(INPUT_LLM)

    print(f"Corpus       : {len(corpus)} entries")
    print(f"Annotator A  : {len(annot_a)} entries")
    print(f"Annotator B  : {len(annot_b)} entries")
    print(f"LLM          : {len(llm)} entries")

    human_labels = merge_human_labels(annot_a, annot_b)
    full         = build_full_corpus(corpus, human_labels, llm)

    print(f"\n=== Violence score distribution ===")
    print(full["violence_score"].value_counts().sort_index())

    yearly = build_d2_yearly(full)

    print(f"\n=== Yearly D2 aggregation ===")
    print(
        yearly[[
            "year", "entry_count", "violence_flag_rate",
            "mean_violence_score", "d2_violence_proportion",
        ]].to_string(index=False)
    )

    OUTPUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT_FULL,   index=False)
    yearly.to_csv(OUTPUT_D2,   index=False)

    print(f"\nSaved full corpus : {OUTPUT_FULL}")
    print(f"Saved D2 yearly   : {OUTPUT_D2}")

    print("""
D2 regression variables:
  d2_violence_proportion  -- share of violent entries per year (primary D2)
  d2_weighted             -- entry_count x mean_violence_score
  mean_violence_score     -- average severity score per year
""")


if __name__ == "__main__":
    main()
