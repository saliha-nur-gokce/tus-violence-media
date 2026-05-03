# CONTEXT.md — Project Guide for Claude Code

Read this file before making any changes to the codebase.

---

## Research Overview

**Question:** Does the intensity of healthcare violence coverage in Turkish media associate
with shifts in TUS branch-level minimum scores?

**Outcome:** `score_min` — TUS (Tıpta Uzmanlık Sınavı) minimum score per specialty per period (2013–2025, 26 periods, ~40 branches).

**Treatment variables:**
- `d1_count` — Annual Google News article count on healthcare violence (primary D1)
- `d2_annot` — Annual proportion of violence-flagged Ekşi Sözlük entries via LLM annotation (primary D2)

**Exposure groups:**
- `high` — Acil Tıp, Genel Cerrahi, Beyin ve Sinir Cerrahisi, Kalp ve Damar Cerrahisi, Göğüs Cerrahisi, Kadın Hastalıkları ve Doğum, Çocuk Cerrahisi, Üroloji, Ortopedi ve Travmatoloji
- `low` — Deri ve Zührevi Hastalıkları, Radyoloji, Nükleer Tıp, Tıbbi Patoloji, Tıbbi Biyokimya, Anatomi, Histoloji ve Embriyoloji, Fizyoloji, Tıbbi Farmakoloji
- `intermediate` — everything else
- **Excluded:** Plastik, Rekonstrüktif ve Estetik Cerrahi (unique selection dynamics)

**Core hypothesis:** D1/D2 negatively associate with high-exposure scores, no effect on low-exposure (reference group).

**Results are reported as correlational** — no causal claims. Following domain expert guidance (Burçay Erus, Boğaziçi University Economics).

---

## Key Analytical Decisions

- `d1_count` chosen over `d1_keyword_mean` and `d1_bert_mean` — only version showing theoretical alignment (high-exposure Spearman r = −0.908)
- `d2_annot` chosen over `d2_severity` and `d2_keyword` — d2_keyword showed no signal, d2_severity has borderline falsification concern (low-exposure p=0.052)
- `post_reform` dummy (≥2022) retained — without it, D1 main effect becomes insignificant (p=0.602) while interactions stay stable
- CPI removed from model
- Year dummy retained as primary spec (over quadratic trend) — interaction coefficients identical across both, year dummy more flexible
- D1 main effect sign instability documented as limitation — not an outlier issue (leave-one-out confirmed), it's a trend-control identification problem
- D2 reported as secondary in paper — global panel signal but weak group-specific differentiation in PanelOLS FE

---

## Current Folder Structure (Local)

This is the current state BEFORE repo cleanup. Use this to locate files.

```
CSSM 530/
│
├── tus_code/
│   └── tus_pdf_parser.py
│       Parses ÖSYM TUS PDFs into panel dataset.
│       Output: tus_panel_macro.csv
│
│       PDF format notes (important for debugging):
│       - 2023–2025: same format
│       - 2022: same as above, minor row differences
│       - 2019–2021: same format, separate columns for general/foreign students
│       - 2018: general and foreign students in separate PDFs → only general used from here back
│       - 2017: table format changes, then stable going back
│       - 2016–2013: p1/p2 splits per year; 2013 also has a separate Sağlık Bilimleri PDF → omitted
│
├── news_code/
│   ├── google_news_retry.py (or .ipynb)
│   │   Scrapes Google News RSS for healthcare violence articles.
│   │   Computes D1 article count.
│   │   Validates against Beyaz Kod data (Spearman r = 0.786, p = 0.036).
│   │   Output: gnews_scored.csv
│   │
│   ├── news_sentiment.py (or .ipynb)
│   │   Merges gnews_scored.csv with BERT yearly scores.
│   │   Computes d1_keyword_weighted, d1_bert_weighted.
│   │   Output: gnews_d1_sentiment_final.csv  ← PRIMARY D1 FILE
│   │
│   └── [BERT script currently on Google Colab — needs to be added here]
│       Input: gnews_scored.csv (from Drive)
│       Output: gnews_bert_scored.csv, gnews_yearly_bert_sentiment.csv
│       Model: savasy/bert-base-turkish-sentiment-cased
│
├── nlp_code/
│   └── eksi_final.py (or .ipynb)
│       Scrapes Ekşi Sözlük healthcare violence threads.
│       Applies keyword-based violence scoring (legacy — not primary D2).
│       Output: eksi_corpus_final.csv  ← RAW EKSI DATA
│
├── llm_annotation/
│   ├── llm_annotation.ipynb
│   │   LLM annotation pipeline (Claude Haiku, few-shot).
│   │   Inputs:
│   │     - eksi_corpus_final.csv
│   │     - eksi_annotation_v2_labeled_[annotator1].csv  ← human labels (Saliha)
│   │     - eksi_annotation_v2_labeled_[annotator2].csv  ← human labels (Emine)
│   │   Output: eksi_llm_annotated.csv
│   │   Inter-annotator κ: physical=0.592, verbal=0.570, anti-violence=0.730
│   │   Note: few-shot examples for violence_normalized and sarcasm use
│   │   annotator1 labels only (agreed positives insufficient) — document this.
│   │
│   └── regression_with_new_d1_d2.ipynb
│       Primary regression notebook.
│       Inputs:
│         - tus_panel_macro.csv
│         - gnews_d1_sentiment_final.csv
│         - eksi_d2_yearly.csv
│       Contains: Spearman, year-level OLS (quadratic + year dummy), PanelOLS branch FE
│       PRIMARY RESULTS FILE — clean this carefully.
│
└── regression/
    └── betterRegressionHope.ipynb (or similar)
        Earlier regression attempts — superseded by llm_annotation/regression_with_new_d1_d2.ipynb
        Keep for reference or move to exploration/.
```

---

## Target Repo Structure (After Cleanup)

```
tus-violence-media/
├── README.md
├── CONTEXT.md                        ← this file
├── requirements.txt
│
├── src/
│   ├── 01_tus_parse.py               ← from tus_code/tus_pdf_parser.py
│   ├── 02_gnews_scrape_validate.py   ← from news_code/google_news_retry
│   ├── 03_gnews_bert_sentiment.ipynb ← Colab notebook (add from Drive)
│   ├── 04_gnews_merge.py             ← from news_code/news_sentiment
│   ├── 05_eksi_scrape.py             ← from nlp_code/eksi_final
│   ├── 06_llm_annotation.ipynb       ← from llm_annotation/llm_annotation.ipynb
│   ├── 07_merge_severity.py          ← merge + D2 yearly aggregation step
│   └── 08_regression.ipynb           ← from llm_annotation/regression_with_new_d1_d2.ipynb
│
├── exploration/
│   ├── d1_version_comparison.py
│   ├── d2_version_comparison.py
│   └── outlier_year_test.py          ← already generated
│
└── codebook/
    └── eksi_annotation_guide_v2.pdf  ← annotation guidelines
```

---

## Cleanup Priorities

When cleaning files, do these in order:

1. **Remove personal names from filenames and code**
   - `eksi_annotation_v2_labeled_saliha.csv` → `eksi_annotation_human_annotator1.csv`
   - `eksi_annotation_v2_labeled_emine.csv` → `eksi_annotation_human_annotator2.csv`
   - Update all references inside notebooks accordingly

2. **Genericize paths**
   - Replace `/Users/salihanurgokce/...` with `DATA_DIR` / `OUTPUT_DIR` variables
   - Add path config block at top of each script

3. **Add markdown cell to each notebook**
   - First cell: what this notebook does, inputs, outputs, pipeline step number

4. **regression_with_new_d1_d2.ipynb — specific issues**
   - Interaction variable naming inconsistency: some cells use `d1_x_high`, others use `d1_count_z_x_high` — standardize to `d1_count_z_x_high` throughout
   - HIGH_EXPOSURE / LOW_EXPOSURE / EXCLUDE lists are repeated in multiple cells — define once at top, reference everywhere
   - Remove any cells from earlier regression attempts that are superseded

5. **Generate requirements.txt**
   - Scan all imports across src/ files
   - Pin versions

---

## Files NOT to Include in Repo

- Raw CSV data files (eksi, tus, gnews)
- Human annotation CSV files (available upon request)
- Any file containing personal paths or names before cleanup
- `regression/betterRegressionHope.ipynb` — superseded, unless moved to exploration/
- Failed Ekşi approach scripts (LDA, BERT binary) — not recoverable, omit
