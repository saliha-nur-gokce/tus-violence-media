# tus-violence-media

**Does healthcare violence coverage affect medical specialty choice in Turkey?**

This repository contains the full analysis pipeline for a correlational study examining whether the intensity of violence-related media coverage and social media discourse associates with shifts in TUS (Tıpta Uzmanlık Sınavı) branch-level minimum scores — the primary revealed-preference indicator of specialty attractiveness among medical residents in Turkey (2013–2025).

Submitted as SP5 for CSSM 530: Automated Text Processing for Social Sciences, Koç University, Spring 2026.

**Domain expert:** Burçay Erus, Associate Professor, Department of Economics, Boğaziçi University.

---

## Research Design

Two treatment variables are constructed and tested against TUS minimum scores across 40 specialties (excluding Plastik Cerrahi), classified into high-, intermediate-, and low-exposure groups based on patient-facing violence risk:

| Variable | Source | Measure |
|----------|--------|---------|
| D1 | Google News RSS | Annual article count on healthcare violence |
| D2 | Ekşi Sözlük | LLM-annotated violence discourse proportion |

Core hypothesis: D1 and D2 should negatively associate with high-exposure specialty scores and show no effect on low-exposure specialties. Tested via Spearman correlations and year-level OLS with interaction terms (D1×exposure group), with PanelOLS branch FE as robustness check.

---

## Pipeline

Run scripts in the following order. Each step's output feeds the next.

```
01_tus_parse.py
    Input : ÖSYM TUS result PDFs (2013–2025, 26 periods)
    Output: tus_panel_macro.csv

02_gnews_scrape_validate.py
    Input : Google News RSS feed
    Output: gnews_scored.csv
            (includes D1 article count + Beyaz Kod validation)

03_gnews_bert_sentiment.ipynb         ← Run on Google Colab (GPU required)
    Input : gnews_scored.csv (from Google Drive)
    Output: gnews_bert_scored.csv
            gnews_yearly_bert_sentiment.csv

04_gnews_merge.py
    Input : gnews_scored.csv
            gnews_yearly_bert_sentiment.csv
    Output: gnews_d1_sentiment_final.csv   ← primary D1 file

05_eksi_scrape.py
    Input : Ekşi Sözlük thread URLs (healthcare violence threads, 2013–2025)
    Output: eksi_corpus_final.csv          (4,148 entries, 54 threads)

06_llm_annotation.ipynb
    Input : eksi_corpus_final.csv
            eksi_annotation_human_annotator1.csv    ← human annotations
            eksi_annotation_human_annotator2.csv    ← human annotations
    Output: eksi_llm_annotated.csv
            (Claude Haiku, few-shot; human κ: physical=0.592, verbal=0.570, anti-violence=0.730)

07_merge_severity.py
    Input : eksi_corpus_final.csv
            eksi_annotation_human_annotator1.csv
            eksi_annotation_human_annotator2.csv
            eksi_llm_annotated.csv
    Output: eksi_annotated_full.csv
            eksi_d2_yearly.csv             ← primary D2 file

08_regression.ipynb
    Input : tus_panel_macro.csv
            gnews_d1_sentiment_final.csv
            eksi_d2_yearly.csv
    Output: sp5_regression_results.csv
            sp5_summary_stats.csv
            figures: parallel_trends.png, d1_d2_trends.png, quota_trend.png
```

---

## Exploration

The `exploration/` folder contains analysis scripts that were run during development but are not part of the main pipeline:

| File | Purpose |
|------|---------|
| `d1_version_comparison.py` | Compares d1_count, d1_keyword_mean, d1_bert_mean — motivates choice of d1_count as primary |
| `d2_version_comparison.py` | Compares d2_annot, d2_severity, d2_keyword — motivates choice of d2_annot as primary |
| `outlier_year_test.py` | Leave-one-out analysis of D1 annual values — confirms interaction coefficients are stable across all year exclusions |

---

## Data

Raw data files are **not included** in this repository.

| Dataset | Source | Availability |
|---------|--------|--------------|
| TUS minimum scores | ÖSYM official PDFs | Publicly available at osym.gov.tr |
| Google News articles | Google News RSS | Reconstructible via `02_gnews_scrape_validate.py` |
| Ekşi Sözlük entries | Ekşi Sözlük (public threads) | Available upon request |
| Human annotation files | Manual coding (two annotators) | Available upon request |

To request data files, open a GitHub issue or contact via email.

---

## D2 Construction: Methodological Iterations

Several approaches were tested before settling on LLM annotation for D2. LDA topic modeling produced low coherence scores. BERT-based binary classification collapsed entries without sufficient discrimination between violence types. Keyword scoring produced no significant signal in any exposure group. Following instructor feedback (Dr. Ali Hürriyetoğlu), human annotation was adopted to establish ground truth, with Claude Haiku (few-shot) extending labels to the full corpus. See `exploration/d2_version_comparison.py` for a quantitative comparison across D2 versions.

---

## Setup

```bash
pip install -r requirements.txt
```

Step 03 (`gnews_bert_sentiment.ipynb`) requires Google Colab with GPU runtime (T4 is sufficient). Upload `gnews_scored.csv` to Google Drive and update `DRIVE_FOLDER` path in the notebook before running.

---

## Requirements

See `requirements.txt`. Key dependencies:

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical operations |
| statsmodels | ≥0.14 | OLS regression |
| linearmodels | ≥5.3 | PanelOLS with entity FE |
| scipy | ≥1.10 | Spearman correlations |
| anthropic | ≥0.25 | LLM annotation (Claude Haiku) |
| transformers | ≥4.35 | BERT sentiment (Colab only) |
| torch | ≥2.0 | BERT inference (Colab only) |
| matplotlib | ≥3.7 | Figures |

---

## Key Results

D1 and D2 show no global effect on TUS scores (D1 p=0.412, D2 p=0.183 without interaction terms). Once exposure-group interactions are added, both variables show consistent negative differential effects on high- and intermediate-exposure specialties relative to low-exposure, across Spearman correlations, year-level OLS, and PanelOLS branch FE. Results are reported as correlational following domain expert guidance.

---

## Reference

Bo, S., Chen, J., Song, Y., & Zhou, S. (2020). Media attention and choice of major: Evidence from anti-doctor violence in China. *Journal of Economic Behavior & Organization*, 170, 1–19. Used as primary methodological reference; results are reported as correlational following domain expert guidance.
