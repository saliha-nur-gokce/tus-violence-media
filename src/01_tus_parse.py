"""
Pipeline step 01 — TUS PDF Parser
Input  : ÖSYM TUS result PDFs (2013–2025, 26 periods) organised in three subfolders
Output : tus_panel_macro.csv  — branch-level program rows with score_min, score_max

Three PDF format groups require separate parsers:
  Layout A  2022–2025 : 8-col, single program-name column, quota_type as own column
  Layout B  2019–2021 : wide format, Genel + Yabancı Uyruk side by side (13 cols)
  Layout C  2013–2018 : 10-col (2013–2014) or 8-col (2015–2018) with Puan Türü column

PDF format notes (important for debugging):
  2023–2025 : same format
  2022      : same as above, minor row differences
  2019–2021 : separate columns for general / foreign students
  2018      : general and foreign students in separate PDFs → only general used from here back
  2017      : table format changes, then stable going back
  2016–2013 : p1/p2 splits per year; 2013 also has a separate Sağlık Bilimleri PDF → omitted
"""

import pdfplumber
import pandas as pd
import re
import sys
from pathlib import Path

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
# Set these paths to match your local directory layout before running.
DATA_DIR = Path("data/raw/tus")          # root directory containing PDF subfolders

PDF_DIR_2022_2025 = DATA_DIR / "2022_2025"
PDF_DIR_2019_2021 = DATA_DIR / "2019_2021"
PDF_DIR_2013_2018 = DATA_DIR / "2013_2018"

OUTPUT_DIR   = Path("data/processed")
OUTPUT_FINAL = OUTPUT_DIR / "tus_panel_macro.csv"
# ─────────────────────────────────────────────────────────────────────────────


# ── BRANCH NORMALISATION ──────────────────────────────────────────────────────
TR_LOWER = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")

# Canonical display names keyed by lowercase-normalised raw string.
# Covers all spelling variants encountered across the 2013–2025 PDFs.
BRANCH_MAP = {
    "acil tip": "Acil Tıp",
    "acil tıp": "Acil Tıp",
    "adli tip": "Adli Tıp",
    "adli tıp": "Adli Tıp",
    "aile hekimligi": "Aile Hekimliği",
    "aile hekimliği": "Aile Hekimliği",
    "anatomi": "Anatomi",
    "anesteziyoloji ve reanimasyon": "Anesteziyoloji ve Reanimasyon",
    "anestezi̇yoloji̇ ve reani̇masyon": "Anesteziyoloji ve Reanimasyon",
    "askeri saglik hizmetleri": "Askeri Sağlık Hizmetleri",
    "askeri sağlık hizmetleri": "Askeri Sağlık Hizmetleri",
    "beyin ve sinir cerrahisi": "Beyin ve Sinir Cerrahisi",
    "beyin ve sinir cerra": "Beyin ve Sinir Cerrahisi",      # truncated variant in older PDFs
    "cocuk cerrahisi": "Çocuk Cerrahisi",
    "çocuk cerrahisi": "Çocuk Cerrahisi",
    "cocuk sagligi ve hastaliklari": "Çocuk Sağlığı ve Hastalıkları",
    "çocuk sağlığı ve hastalıkları": "Çocuk Sağlığı ve Hastalıkları",
    "çocuk sağlığı ve hast": "Çocuk Sağlığı ve Hastalıkları",
    "cocuk ve ergen ruh sagligi ve hastaliklari": "Çocuk ve Ergen Ruh Sağlığı ve Hastalıkları",
    "çocuk ve ergen ruh sağlığı ve hastalıkları": "Çocuk ve Ergen Ruh Sağlığı ve Hastalıkları",
    "çocuk ve ergen ru": "Çocuk ve Ergen Ruh Sağlığı ve Hastalıkları",
    "çocuk ve ergen ruh sağlığı v": "Çocuk ve Ergen Ruh Sağlığı ve Hastalıkları",
    "deri ve zuhrevi hastaliklari": "Deri ve Zührevi Hastalıkları",
    "deri ve zührevi hastalıkları": "Deri ve Zührevi Hastalıkları",
    "enfeksiyon hastaliklari ve klinik mikrobiyoloji": "Enfeksiyon Hastalıkları ve Klinik Mikrobiyoloji",
    "enfeksiyon hastalıkları ve klinik mikrobiyoloji": "Enfeksiyon Hastalıkları ve Klinik Mikrobiyoloji",
    "enfeksiyon hastaliklari ve klinik mik": "Enfeksiyon Hastalıkları ve Klinik Mikrobiyoloji",
    "enfeksiyon hastaliklari ve klinik mikro": "Enfeksiyon Hastalıkları ve Klinik Mikrobiyoloji",
    "fiziksel tip ve rehabilitasyon": "Fiziksel Tıp ve Rehabilitasyon",
    "fiziksel tıp ve rehabilitasyon": "Fiziksel Tıp ve Rehabilitasyon",
    "fi̇zi̇ksel tip ve rehabi̇li̇tasyon": "Fiziksel Tıp ve Rehabilitasyon",
    "fizyoloji": "Fizyoloji",
    "genel cerrahi": "Genel Cerrahi",
    "gogus cerrahisi": "Göğüs Cerrahisi",
    "göğüs cerrahisi": "Göğüs Cerrahisi",
    "gogus hastaliklari": "Göğüs Hastalıkları",
    "göğüs hastalıkları": "Göğüs Hastalıkları",
    "goz hastaliklari": "Göz Hastalıkları",
    "göz hastalıkları": "Göz Hastalıkları",
    "halk sagligi": "Halk Sağlığı",
    "halk sağlığı": "Halk Sağlığı",
    "hava ve uzay hekimligi": "Hava ve Uzay Hekimliği",
    "hava ve uzay hekimliği": "Hava ve Uzay Hekimliği",
    "histoloji ve embriyoloji": "Histoloji ve Embriyoloji",
    "ic hastaliklari": "İç Hastalıkları",
    "iç hastalıkları": "İç Hastalıkları",
    "kadin hastaliklari ve dogum": "Kadın Hastalıkları ve Doğum",
    "kadın hastalıkları ve doğum": "Kadın Hastalıkları ve Doğum",
    "kadin hastaliklari ve": "Kadın Hastalıkları ve Doğum",
    "kalp ve damar cerrahisi": "Kalp ve Damar Cerrahisi",
    "kardiyoloji": "Kardiyoloji",
    "kulak burun bogaz hastaliklari": "Kulak Burun Boğaz Hastalıkları",
    "kulak burun boğaz hastalıkları": "Kulak Burun Boğaz Hastalıkları",
    "noroloji": "Nöroloji",
    "nöroloji": "Nöroloji",
    "nukleer tip": "Nükleer Tıp",
    "nükleer tıp": "Nükleer Tıp",
    "ortopedi ve travmatoloji": "Ortopedi ve Travmatoloji",
    "plastik, rekonstruktif ve estetik cerrahi": "Plastik, Rekonstrüktif ve Estetik Cerrahi",
    "plastik, rekonstrüktif ve estetik cerrahi": "Plastik, Rekonstrüktif ve Estetik Cerrahi",
    "radyasyon onkolojisi": "Radyasyon Onkolojisi",
    "radyoloji": "Radyoloji",
    "ruh sagligi ve hastaliklari": "Ruh Sağlığı ve Hastalıkları",
    "ruh sağlığı ve hastalıkları": "Ruh Sağlığı ve Hastalıkları",
    "ruh sagligi ve hast": "Ruh Sağlığı ve Hastalıkları",
    "spor hekimligi": "Spor Hekimliği",
    "spor hekimliği": "Spor Hekimliği",
    "sualtı hekimliği ve hiperbarik tıp": "Sualtı Hekimliği ve Hiperbarik Tıp",
    "sualtı hekimligi ve hiperbarik tıp": "Sualtı Hekimliği ve Hiperbarik Tıp",
    "su altı hekimliği ve hiperbarik tıp": "Sualtı Hekimliği ve Hiperbarik Tıp",
    "su alti hekimligi ve hiperbarik tip": "Sualtı Hekimliği ve Hiperbarik Tıp",
    "su alti heki̇mli̇ği̇ ve hi̇perbari̇k tip": "Sualtı Hekimliği ve Hiperbarik Tıp",
    "tibbi biyokimya": "Tıbbi Biyokimya",
    "tıbbi biyokimya": "Tıbbi Biyokimya",
    "tıbbi ekoloji ve hidroklimatoloji": "Tıbbi Ekoloji ve Hidroklimatoloji",
    "tibbi ekoloji ve hidroklimatoloji": "Tıbbi Ekoloji ve Hidroklimatoloji",
    "tıbbi ekoloji ve hi̇drokli̇matoloji̇": "Tıbbi Ekoloji ve Hidroklimatoloji",
    "tibbi farmakoloji": "Tıbbi Farmakoloji",
    "tıbbi farmakoloji": "Tıbbi Farmakoloji",
    "tibbi genetik": "Tıbbi Genetik",
    "tıbbi genetik": "Tıbbi Genetik",
    "tibbi mikrobiyoloji": "Tıbbi Mikrobiyoloji",
    "tıbbi mikrobiyoloji": "Tıbbi Mikrobiyoloji",
    "tibbi patoloji": "Tıbbi Patoloji",
    "tıbbi patoloji": "Tıbbi Patoloji",
    "uroloji": "Üroloji",
    "üroloji": "Üroloji",
}

# Rows with these branch suffixes represent special quota types — drop them.
_SPECIAL_SUFFIX_RE = re.compile(
    r"\(K\.K\.T\.C\..*?\)|\(M\.A\.P\.\)|\(T\.D\.M\.M\.İ\)",
    re.IGNORECASE,
)

OUTPUT_COLS = [
    "year", "period", "program_code", "institution", "branch",
    "quota_count", "placed_count", "empty_quota",
    "score_min", "score_max", "filled",
]


def _normalize_branch(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return raw
    cleaned = re.sub(r"\s+", " ", raw.strip())
    key = cleaned.translate(TR_LOWER).lower()
    return BRANCH_MAP.get(key, cleaned)


def _parse_score(v: str) -> float:
    if not v or str(v).strip() in ("--", "-", "—", ""):
        return float("nan")
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _to_int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _to_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["filled"] = (df["placed_count"].fillna(0) > 0).astype(int)
    for col in ["quota_count", "placed_count", "empty_quota"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df[OUTPUT_COLS]


# ── LAYOUT A  2022–2025 ───────────────────────────────────────────────────────
_QUOTA_TYPES_A = {"Genel", "Yabancı Uyruk", "Yabanci Uyruk"}


def _parse_pdf_layout_a(pdf_path: Path, year: int, period: int) -> pd.DataFrame:
    """Parse 2022–2025 PDFs (Layout A: 8-col, quota_type as own column)."""
    rows = []
    unknown_branches: set = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                for row in table:
                    if not row or len(row) < 6:
                        continue
                    r = [str(c).replace("\n", " ").strip() if c else "" for c in row]

                    if not re.match(r"^\d{6,}", r[0]):
                        continue

                    code, program = r[0], r[1]

                    if r[2] in _QUOTA_TYPES_A:
                        qt = r[2].replace("Yabanci Uyruk", "Yabancı Uyruk")
                        quota_s, placed_s, empty_s = r[3], r[4], r[5]
                        smin_s = r[6] if len(r) > 6 else ""
                        smax_s = r[7] if len(r) > 7 else ""
                    else:
                        m = re.search(r"\s+(Genel|Yabancı Uyruk|Yabanci Uyruk)$", program)
                        if m:
                            qt = m.group(1).replace("Yabanci Uyruk", "Yabancı Uyruk")
                            program = program[: m.start()].strip()
                        else:
                            qt = None
                        quota_s, placed_s, empty_s = r[2], r[3], r[4]
                        smin_s = r[5] if len(r) > 5 else ""
                        smax_s = r[6] if len(r) > 6 else ""

                    if qt != "Genel":
                        continue

                    if "/" in program:
                        inst, br_raw = program.rsplit("/", 1)
                        inst = re.sub(r"\s+", " ", inst.strip())
                        br_raw = br_raw.strip()
                        branch = _normalize_branch(br_raw)
                        if br_raw.translate(TR_LOWER).lower() not in BRANCH_MAP:
                            unknown_branches.add(br_raw)
                    else:
                        inst, branch = program.strip(), None

                    rows.append({
                        "year": year, "period": period,
                        "program_code": code,
                        "institution": inst, "branch": branch,
                        "quota_count": _to_int(quota_s),
                        "placed_count": _to_int(placed_s),
                        "empty_quota": _to_int(empty_s),
                        "score_min": _parse_score(smin_s),
                        "score_max": _parse_score(smax_s),
                    })

    if not rows:
        print(f"  WARNING: no data rows — {pdf_path.name}")
        return pd.DataFrame()
    if unknown_branches:
        print(f"  [WARN] unmapped branches ({pdf_path.name}): {sorted(unknown_branches)}")
    return _to_dataframe(rows)


# ── LAYOUT B  2019–2021 ───────────────────────────────────────────────────────
def _parse_pdf_layout_b(pdf_path: Path, year: int, period: int) -> pd.DataFrame:
    """
    Parse 2019–2021 PDFs (Layout B: wide format).

    Column order per data row (13+ cols):
      [0] dal_kodu  [1] dal_adi  [2] tablo (K/T — ignored)
      [3] genel_kontenjan  [4] genel_yerlesen  [5] genel_bos
      [6] genel_min  [7] genel_max
      [8-12] yabanci block — ignored
    """
    rows = []
    unknown_branches: set = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                for row in table:
                    if not row or len(row) < 6:
                        continue
                    r = [str(c).replace("\n", " ").strip() if c else "" for c in row]

                    if not re.match(r"^\d{6,}", r[0]):
                        continue

                    code   = r[0]
                    dal_adi = r[1]
                    quota_s  = r[3] if len(r) > 3 else ""
                    placed_s = r[4] if len(r) > 4 else ""
                    empty_s  = r[5] if len(r) > 5 else ""
                    smin_s   = r[6] if len(r) > 6 else ""
                    smax_s   = r[7] if len(r) > 7 else ""

                    if "/" in dal_adi:
                        inst, br_raw = dal_adi.rsplit("/", 1)
                        inst = re.sub(r"\s+", " ", inst.strip())
                        br_raw = br_raw.strip()
                        branch = _normalize_branch(br_raw)
                        if br_raw.translate(TR_LOWER).lower() not in BRANCH_MAP:
                            unknown_branches.add(br_raw)
                    else:
                        inst = re.sub(r"\s+", " ", dal_adi.strip())
                        branch = None

                    rows.append({
                        "year": year, "period": period,
                        "program_code": code,
                        "institution": inst, "branch": branch,
                        "quota_count": _to_int(quota_s),
                        "placed_count": _to_int(placed_s),
                        "empty_quota": _to_int(empty_s),
                        "score_min": _parse_score(smin_s),
                        "score_max": _parse_score(smax_s),
                    })

    if not rows:
        print(f"  WARNING: no data rows — {pdf_path.name}")
        return pd.DataFrame()
    if unknown_branches:
        print(f"  [WARN] unmapped branches ({pdf_path.name}): {sorted(unknown_branches)}")
    return _to_dataframe(rows)


# ── LAYOUT C  2013–2018 ───────────────────────────────────────────────────────
_PUAN_TURU = {"k", "t"}


def _parse_pdf_layout_c(pdf_path: Path, year: int, period: int) -> pd.DataFrame:
    """
    Parse 2013–2018 PDFs (Layout C).

    Auto-detects sub-layout:
      10-col (2013–2014): col[4] is Puan Türü; Kurum, Fakülte, Branş as separate cols
      8-col  (2015–2018): col[2] is Puan Türü; Dal Adı as single col
    """
    rows = []
    unknown_branches: set = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                for row in table:
                    if not row or len(row) < 6:
                        continue
                    r = [str(c).replace("\n", " ").strip() if c else "" for c in row]

                    if not re.match(r"^\d{6,}", r[0]):
                        continue

                    code = r[0]

                    if len(r) >= 10 and r[4].strip().lower() in _PUAN_TURU:
                        # 10-col layout: institution and faculty in separate columns
                        inst   = re.sub(r"\s+", " ", (r[1] + "/" + r[2]).strip("/ "))
                        br_raw = r[3].strip()
                        quota_s, placed_s, empty_s = r[5], r[6], r[7]
                        smin_s = r[8] if len(r) > 8 else ""
                        smax_s = r[9] if len(r) > 9 else ""

                    elif len(r) >= 8 and r[2].strip().lower() in _PUAN_TURU:
                        # 8-col layout: single Dal Adı column
                        dal_adi = r[1]
                        if "/" in dal_adi:
                            inst, br_raw = dal_adi.rsplit("/", 1)
                            inst = re.sub(r"\s+", " ", inst.strip())
                            br_raw = br_raw.strip()
                        else:
                            inst, br_raw = dal_adi.strip(), ""
                        quota_s, placed_s, empty_s = r[3], r[4], r[5]
                        smin_s = r[6] if len(r) > 6 else ""
                        smax_s = r[7] if len(r) > 7 else ""

                    else:
                        continue

                    if br_raw and _SPECIAL_SUFFIX_RE.search(br_raw):
                        continue

                    branch = _normalize_branch(br_raw) if br_raw else None
                    if br_raw and br_raw.translate(TR_LOWER).lower() not in BRANCH_MAP:
                        unknown_branches.add(br_raw)

                    rows.append({
                        "year": year, "period": period,
                        "program_code": code,
                        "institution": inst, "branch": branch,
                        "quota_count": _to_int(quota_s),
                        "placed_count": _to_int(placed_s),
                        "empty_quota": _to_int(empty_s),
                        "score_min": _parse_score(smin_s),
                        "score_max": _parse_score(smax_s),
                    })

    if not rows:
        print(f"  WARNING: no data rows — {pdf_path.name}")
        return pd.DataFrame()
    if unknown_branches:
        print(f"  [WARN] unmapped branches ({pdf_path.name}): {sorted(unknown_branches)}")
    return _to_dataframe(rows)


# ── DIRECTORY PARSER ──────────────────────────────────────────────────────────
_FNAME_RE = re.compile(r"tus_(\d{4})_p(\d+)\.pdf", re.IGNORECASE)


def parse_directory(pdf_dir: Path, layout_fn, label: str) -> pd.DataFrame:
    files = sorted(pdf_dir.glob("tus_*.pdf"))
    if not files:
        print(f"  No tus_*.pdf files found in {pdf_dir}")
        return pd.DataFrame()

    all_dfs = []
    for f in files:
        m = _FNAME_RE.match(f.name)
        if not m:
            print(f"  Skipping {f.name} (name does not match tus_YYYY_pN.pdf)")
            continue
        year, period = int(m.group(1)), int(m.group(2))
        print(f"  [{label}] {f.name}  year={year} period={period} ...", end="", flush=True)
        df = layout_fn(f, year, period)
        if df.empty:
            continue
        print(f"  {len(df)} rows")
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    panel = pd.concat(all_dfs, ignore_index=True)
    return panel.sort_values(["year", "period", "institution", "branch"]).reset_index(drop=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Parsing Layout A (2022–2025) ===")
    df_a = parse_directory(PDF_DIR_2022_2025, _parse_pdf_layout_a, "A")

    print("\n=== Parsing Layout B (2019–2021) ===")
    df_b = parse_directory(PDF_DIR_2019_2021, _parse_pdf_layout_b, "B")

    print("\n=== Parsing Layout C (2013–2018) ===")
    df_c = parse_directory(PDF_DIR_2013_2018, _parse_pdf_layout_c, "C")

    panels = [d for d in [df_a, df_b, df_c] if not d.empty]
    if not panels:
        print("Nothing parsed — check PDF_DIR paths in the config block.")
        sys.exit(1)

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["year", "period", "institution", "branch"]).reset_index(drop=True)

    # Fix residual branch variants not caught by BRANCH_MAP, and drop out-of-scope branches.
    branch_fix = {
        "SU ALTI HEKİMLİĞİ VE HİPERBARİK TIP": "Sualtı Hekimliği ve Hiperbarik Tıp",
        "Su Altı Hekimliği Ve Hiperbarik Tıp":  "Sualtı Hekimliği ve Hiperbarik Tıp",
        "TIBBİ EKOLOJİ VE HİDROKLİMATOLOJİ":   "Tıbbi Ekoloji ve Hidroklimatoloji",
        "AĞIZ, YÜZ VE ÇENE CERRAHİSİ":          None,  # not tracked in this study
    }
    panel["branch"] = panel["branch"].replace(branch_fix)
    panel = panel[panel["branch"].notna()].copy()

    before = len(panel)
    panel = panel[panel["quota_count"].notna()].copy()
    print(f"\nDropped {before - len(panel)} rows with no quota data (foreign-student placeholders)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"Output   : {OUTPUT_FINAL}")
    print(f"Rows     : {len(panel):,}")
    print(f"Years    : {sorted(panel['year'].unique())}")
    print(f"Branches : {panel['branch'].nunique()}")
    print(f"NaN score_min : {panel['score_min'].isna().sum():,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
