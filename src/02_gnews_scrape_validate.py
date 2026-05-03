"""
Pipeline step 02 — Google News RSS Scraper + D1 Validation
Input  : Google News RSS feed (live; no local input file required)
Outputs:
    gnews_scored.csv          — deduplicated article-level data with keyword/group tags
    gnews_yearly_counts.csv   — yearly treatment vs. control article counts
    d1_metrics.csv            — D1 treatment variable (d1_count, d1_binary, d1_keyword_coverage)
    whitecode_d1_merged.csv   — D1 merged with Beyaz Kod figures for trend validation

D1 validation result: Spearman r = 0.786, p = 0.036 (excl. 2018; n=7 years)
Primary D1 variable: d1_count (yearly deduplicated article count, 5 treatment keywords)
"""

import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/raw/gnews")

OUTPUT_ARTICLES   = OUTPUT_DIR / "gnews_scored.csv"
OUTPUT_YEARLY     = OUTPUT_DIR / "gnews_yearly_counts.csv"
OUTPUT_D1         = OUTPUT_DIR / "d1_metrics.csv"
OUTPUT_VALIDATION = OUTPUT_DIR / "whitecode_d1_merged.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── SCRAPER CONFIG ────────────────────────────────────────────────────────────
YEARS = list(range(2013, 2026))

# Treatment: healthcare violence keywords
TREATMENT_KEYWORDS = [
    "doktora şiddet",
    "hekime saldırı",
    "sağlıkta şiddet",
    "sağlık çalışanına şiddet",
    "hastanede şiddet",
    "hekime şiddet",
    "hastaneye saldırı",
    "sağlık çalışanına saldırı",
]

# Control: legal profession (parallel falsification group, following Bo et al. 2020)
CONTROL_KEYWORDS = [
    "avukata saldırı",
    "hakime saldırı",
    "avukata şiddet",
]

ALL_KEYWORDS = TREATMENT_KEYWORDS + CONTROL_KEYWORDS

BASE_URL    = "https://news.google.com/rss/search"
PARAMS_BASE = {"hl": "tr", "gl": "TR", "ceid": "TR:tr"}
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5  # seconds between requests

# ── D1 CLEANING CONFIG ────────────────────────────────────────────────────────
# Keywords excluded from D1 computation due to high off-topic noise
NOISY_KEYWORDS = [
    "hastaneye saldırı",
    "sağlık çalışanına saldırı",
    "sağlıkta şiddet",
]

# Title-level noise patterns: off-topic violence (terrorism, crime, international)
NOISE_PATTERNS = [
    "PKK", "terör saldırı", "bombalı saldırı", "roketatarlı",
    "asker şehit", "cinsel saldırı", "Mısır", "Boston", "Somali",
    "Nijerya", "Afganistan", "tuvalet krizi", "depremde", "sel ",
]

# 5 clean treatment keywords retained for D1 after removing noisy ones
TREATMENT_KEYWORDS_CLEAN = [
    "doktora şiddet",
    "hastanede şiddet",
    "hekime saldırı",
    "hekime şiddet",
    "sağlık çalışanına şiddet",
]

# ── BEYAZ KOD REFERENCE DATA ──────────────────────────────────────────────────
# Manually compiled from press sources.
# 2013-2017: Milliyet April 2018 (Ministry of Health figures)
# 2018:      Torun (2020) Cukurova Medical Journal — physical + verbal combined
# 2019:      Not found
# 2020-2021: TTB / Diken July 2022
# 2022+:     Ministry stopped publishing
WHITE_CODE_DATA = {
    2013: 10715,
    2014: 11174,
    2015: 11881,
    2016: 13076,
    2017: 13545,
    2018: 9108,   # likely physical-violence-only filings — excluded from primary correlation
    2019: None,
    2020: 11942,
    2021: 29826,  # spike: Dr. Ertan İskender stabbing (May 2021) + legislative changes
    2022: None,
    2023: None,
    2024: None,
    2025: None,
}

# ── RSS UTILITIES ─────────────────────────────────────────────────────────────

def _parse_pub_date(pub_date: str) -> tuple:
    """Return (year, iso_date_str) from an RSS pubDate string."""
    if not pub_date:
        return None, None
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %Z",
    ]:
        try:
            dt = datetime.strptime(pub_date.strip(), fmt)
            return dt.year, dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None, None


def _fetch_rss(query: str) -> list:
    params = {**PARAMS_BASE, "q": query}
    try:
        response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"      [ERROR] {e}")
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"      [XML ERROR] {e}")
        return []

    items = []
    for item in root.findall(".//item"):
        title    = item.findtext("title", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        link     = item.findtext("link", "").strip()
        source   = item.findtext("source", "").strip()
        year, date_parsed = _parse_pub_date(pub_date)
        if not title:
            continue
        items.append({
            "title":       title,
            "source":      source,
            "date_raw":    pub_date,
            "date_parsed": date_parsed,
            "year":        year,
            "url":         link,
        })
    return items


def _dedup_articles(articles: list) -> list:
    seen, result = set(), []
    for a in articles:
        key = a["url"] or a["title"]
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result


# ── SCRAPER ───────────────────────────────────────────────────────────────────

def scrape_all_years() -> pd.DataFrame:
    """Scrape Google News RSS for all years and keywords. Returns article-level DataFrame."""
    print("=" * 65)
    print("GOOGLE NEWS FULL HISTORICAL SCRAPER")
    print(f"Years   : {YEARS[0]}–{YEARS[-1]}")
    print(f"Keywords: {len(ALL_KEYWORDS)} total "
          f"({len(TREATMENT_KEYWORDS)} treatment, {len(CONTROL_KEYWORDS)} control)")
    total_req = len(YEARS) * len(ALL_KEYWORDS)
    print(f"Requests: ~{total_req} | Est. time: ~{total_req * REQUEST_DELAY / 60:.1f} min")
    print("=" * 65)

    all_articles = []

    for year in YEARS:
        date_after  = f"{year}-01-01"
        date_before = f"{year + 1}-01-01"
        print(f"\n[YEAR {year}]")
        year_articles = []

        for keyword in ALL_KEYWORDS:
            group = "treatment" if keyword in TREATMENT_KEYWORDS else "control"
            query = f"{keyword} after:{date_after} before:{date_before}"
            items = _fetch_rss(query)
            for item in items:
                item["keyword"]    = keyword
                item["group"]      = group
                item["query_year"] = year
            year_articles.extend(items)
            print(f"  [{group:9s}] '{keyword}': {len(items):2d} articles")
            time.sleep(REQUEST_DELAY)

        year_deduped = _dedup_articles(year_articles)
        t_n = sum(1 for a in year_deduped if a["group"] == "treatment")
        c_n = sum(1 for a in year_deduped if a["group"] == "control")
        print(f"  -> Year {year} deduped: {len(year_deduped)} | treatment: {t_n} | control: {c_n}")
        all_articles.extend(year_deduped)

    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset=["url", "query_year"])
    return df


def build_yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year in YEARS:
        year_df = df[df["query_year"] == year]
        t_dedup = year_df[year_df["group"] == "treatment"].drop_duplicates(subset=["url"])
        c_dedup = year_df[year_df["group"] == "control"].drop_duplicates(subset=["url"])
        records.append({
            "year":                year,
            "treatment_articles":  len(t_dedup),
            "control_articles":    len(c_dedup),
            "total_articles":      len(t_dedup) + len(c_dedup),
        })
    return pd.DataFrame(records)


# ── D1 METRIC BUILDER ─────────────────────────────────────────────────────────

def _is_noise(title: str) -> bool:
    t = title.lower()
    return any(p.lower() in t for p in NOISE_PATTERNS)


def build_d1_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute D1 variables from the cleaned article-level DataFrame.

    d1_count            : yearly deduplicated article count (primary D1)
    d1_binary           : 1 if any article exists that year
    d1_keyword_coverage : number of distinct keywords with >= 1 result (max 5)
    """
    clean = df[df["group"] == "treatment"].copy()
    clean = clean[~clean["keyword"].isin(NOISY_KEYWORDS)]
    clean = clean[~clean["title"].apply(_is_noise)]
    clean["query_year"] = clean["query_year"].astype(int)

    records = []
    for year in sorted(clean["query_year"].unique()):
        year_df = clean[clean["query_year"] == year]
        year_dedup = year_df.drop_duplicates(subset=["url"])
        records.append({
            "year":                year,
            "d1_count":            len(year_dedup),
            "d1_binary":           int(len(year_dedup) > 0),
            "d1_keyword_coverage": year_df["keyword"].nunique(),
        })
    return pd.DataFrame(records)


# ── BEYAZ KOD VALIDATION ──────────────────────────────────────────────────────

def validate_d1_whitecode(d1: pd.DataFrame) -> pd.DataFrame:
    """Merge D1 with Beyaz Kod data and compute Spearman correlation."""
    wc = pd.DataFrame([
        {"year": y, "whitecode_count": v}
        for y, v in WHITE_CODE_DATA.items()
    ])
    merged = pd.merge(d1[["year", "d1_count"]], wc, on="year", how="left")

    print("\n=== D1 vs. BEYAZ KOD ===")
    print(f"{'Year':<6} {'D1 count':>10} {'Beyaz Kod':>12}")
    print("-" * 32)
    for _, row in merged.iterrows():
        wc_val = f"{int(row['whitecode_count']):>12,}" if pd.notna(row["whitecode_count"]) else "         NaN"
        print(f"{int(row['year']):<6} {int(row['d1_count']):>10} {wc_val}")

    valid = merged.dropna(subset=["whitecode_count"])
    valid_excl = valid[valid["year"] != 2018]   # 2018 Beyaz Kod is physical-only

    if len(valid_excl) >= 4:
        sr, sp = spearmanr(valid_excl["d1_count"], valid_excl["whitecode_count"])
        pr, pp = pearsonr(valid_excl["d1_count"], valid_excl["whitecode_count"])
        print(f"\nCorrelation (excl. 2018, n={len(valid_excl)}): "
              f"Spearman r={sr:.3f} p={sp:.3f} | Pearson r={pr:.3f} p={pp:.3f}")

    return merged


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: scrape
    df_articles = scrape_all_years()
    df_articles.to_csv(OUTPUT_ARTICLES, index=False, encoding="utf-8-sig")
    print(f"\nSaved articles : {OUTPUT_ARTICLES} ({len(df_articles)} rows)")

    # Step 2: yearly counts
    df_yearly = build_yearly_counts(df_articles)
    df_yearly.to_csv(OUTPUT_YEARLY, index=False, encoding="utf-8-sig")
    print(f"Saved yearly   : {OUTPUT_YEARLY}")

    # Step 3: D1 metrics
    df_d1 = build_d1_metrics(df_articles)
    df_d1.to_csv(OUTPUT_D1, index=False, encoding="utf-8-sig")
    print(f"Saved D1       : {OUTPUT_D1}")

    # Step 4: Beyaz Kod validation
    df_merged = validate_d1_whitecode(df_d1)
    df_merged.to_csv(OUTPUT_VALIDATION, index=False, encoding="utf-8-sig")
    print(f"Saved validation: {OUTPUT_VALIDATION}")

    print(f"\n{'='*55}")
    print("YEARLY D1 SUMMARY")
    print(f"{'Year':<6} {'d1_count':>10} {'d1_kw_cov':>10}")
    print("-" * 30)
    for _, row in df_d1.iterrows():
        print(f"{int(row['year']):<6} {int(row['d1_count']):>10} {int(row['d1_keyword_coverage']):>10}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
