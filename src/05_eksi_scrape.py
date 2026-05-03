"""
Pipeline step 05 — Eksi Sozluk Corpus Scraper
Input  : none (live web scraping via Selenium)
Outputs:
    eksi_batch_a.csv      — test batch (first 15 threads)
    eksi_batch_b.csv      — remaining threads (full depth)
    eksi_corpus_raw.csv   — merged, deduplicated, before date filter
    eksi_corpus_final.csv — filtered to DATE_CUTOFF_YEAR–2025

Workflow:
    1. Run main()           — runs batch A (test), then batch B, then merges
    Alternatively:
    1. run_batch(BATCH_A, max_pages=2, output_file=OUTPUT_BATCH_A)
    2. Inspect eksi_batch_a.csv
    3. run_batch(BATCH_B, max_pages=10, output_file=OUTPUT_BATCH_B)
    4. merge_and_filter()

Requires: selenium, pandas
    pip install selenium pandas
"""

import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ── PATH CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("data/raw/eksi")
OUTPUT_BATCH_A = OUTPUT_DIR / "eksi_batch_a.csv"
OUTPUT_BATCH_B = OUTPUT_DIR / "eksi_batch_b.csv"
OUTPUT_RAW     = OUTPUT_DIR / "eksi_corpus_raw.csv"
OUTPUT_FINAL   = OUTPUT_DIR / "eksi_corpus_final.csv"
LOG_FILE       = OUTPUT_DIR / "scraper_log.txt"
# ─────────────────────────────────────────────────────────────────────────────

DATE_CUTOFF_YEAR = 2013  # entries before this year are dropped from the final corpus

# ── TARGET THREADS ────────────────────────────────────────────────────────────
# 65 threads spanning violence, migration/resignation, career/preference,
# working conditions/salary, TUS/specialty, and health policy.
TARGET_THREADS = [
    # Violence
    "https://eksisozluk.com/doktora-siddet--5540898",
    "https://eksisozluk.com/hekime-yonelik-siddete-davetiye-cikaran-trt-yayini--5038442",
    "https://eksisozluk.com/hekime-siddetin-en-onemli-nedeni--7171915",
    "https://eksisozluk.com/doktora-dokunmak-konusmak-falan-bu-ne-manyakliktir--7421387",
    "https://eksisozluk.com/6-temmuz-2022-konyada-doktora-silahli-saldiri--7332483",
    "https://eksisozluk.com/doktora-siddetin-cozumu--3349084",
    "https://eksisozluk.com/ersin-arslan--3338605",
    "https://eksisozluk.com/doktora-siddet-haberlerinin-halk-uzerindeki-etkisi--3349061",
    "https://eksisozluk.com/cuma-hutbesinde-doktorlarin-hedef-gosterilmesi--7334696",
    "https://eksisozluk.com/sanliurfada-doktora-parke-tasi-ile-saldiri--5726404",
    "https://eksisozluk.com/doktora-saldiri-haberine-gelen-korkunc-yorumlar--7876426",
    "https://eksisozluk.com/saglik-calisanlarina-yonelik-siddet--3268438",
    "https://eksisozluk.com/saglikta-siddet--3697813",
    "https://eksisozluk.com/saglikciya-hakaret-eden-3-yil-hapis-yatacak--6475559",
    "https://eksisozluk.com/olum-haberi-veren-doktoru-yumruklayan-sahis--7779424",
    # Migration / resignation
    "https://eksisozluk.com/doktorlarin-almanyaya-gocu--5856368",
    "https://eksisozluk.com/hekimler-istifa-ediyor--7100855",
    "https://eksisozluk.com/ulkeyi-terkeden-doktora-soylenmek-istenenler--7149481",
    "https://eksisozluk.com/50-bin-doktorun-yurt-disi-icin-hazirlik-yapmasi--7212759",
    "https://eksisozluk.com/doktorlarin-hedef-haline-gelme-sebebi--7214261",
    "https://eksisozluk.com/doktorlarin-is-birakma-sebebi--7180634",
    "https://eksisozluk.com/14-15-16-mart-2022-doktorlarin-is-birakmasi--7181460",
    "https://eksisozluk.com/haberturk-canli-yayinda-isyan-eden-saglik-calisani--7587886",
    # Career / preference
    "https://eksisozluk.com/tip-fakultesi--33583",
    "https://eksisozluk.com/tip-okumak--131997",
    "https://eksisozluk.com/doktor-olmak-isteyenlere-tavsiyeler--1610647",
    "https://eksisozluk.com/universite-tercihi-yapacaklara-tavsiyeler--1315523",
    "https://eksisozluk.com/tip-okuyan-10-kisiden-9unun-koylu-olmasi--7019916",
    "https://eksisozluk.com/bir-doktorun-egosundan-daha-buyuk-olan-sey--4831966",
    "https://eksisozluk.com/doktorun-biz-malin-onde-gideniyiz-demesi--7279770",
    "https://eksisozluk.com/hemsire--121799",
    "https://eksisozluk.com/hemsirelere-hekimlik-yolu-acilsin--5773002",
    # Working conditions / salary
    "https://eksisozluk.com/doktor--49832",
    "https://eksisozluk.com/doktor-maasi--1819226",
    "https://eksisozluk.com/doktor-maasinin-gereginden-fazla-olmasi--4025050",
    "https://eksisozluk.com/asistan-doktor--403443",
    "https://eksisozluk.com/doktorun-doktora-mobbing-uygulamasi-rezaleti--7095621",
    "https://eksisozluk.com/turk-tabipleri-birligi--1175785",
    "https://eksisozluk.com/saglik-sisteminin-icler-acisi-hali--5223757",
    # TUS / specialty
    "https://eksisozluk.com/tipta-uzmanlik-egitimi-giris-sinavi--1650861",
    "https://eksisozluk.com/18-nisan-2022-tus-kadrolari-rezaleti--7243994",
    "https://eksisozluk.com/26-kasim-2021-tus-ile-ilgili-duzenleme-calismalari--7093771",
    # Health system / policy
    "https://eksisozluk.com/saglik-bakanligi--236712",
    "https://eksisozluk.com/saglik-sistemi--774369",
    "https://eksisozluk.com/aile-hekimi--987216",
    "https://eksisozluk.com/sehir-hastaneleri--3723109",
    "https://eksisozluk.com/hastanelerde-artik-sira-beklenmedigi-gercegi--6693969",
    "https://eksisozluk.com/devletin-hastane-isletmesi-sacmaligi--7333084",
    # Health ministers
    "https://eksisozluk.com/fahrettin-koca--3568141",
    "https://eksisozluk.com/recep-akdag--427433",
    "https://eksisozluk.com/kemal-memisoglu--5694721",
    # Additional threads
    "https://eksisozluk.com/saglikcilarin-24-saat-nobet-tutmasina-karsi-olmak--6095725",
    "https://eksisozluk.com/sosyal-sigortalar-ve-genel-saglik-sigortasi-yasasi--1535080",
    "https://eksisozluk.com/doktor-grevi--839210",
    "https://eksisozluk.com/doktor-grevinin-akpye-yaramasi--7336159",
    "https://eksisozluk.com/27-ekim-2020-saglik-calisanlarina-istifa-yasagi--6715152",
    "https://eksisozluk.com/doktorlarin-bilincli-hastaya-tahammul-edememesi--2466573",
    "https://eksisozluk.com/doktorlarin-hastalara-sen-diye-hitap-etmeleri--1220788",
    "https://eksisozluk.com/aile-hekimligi--223242",
    "https://eksisozluk.com/cerrahi-branslarda-doktor-bulamama-tehlikesi--8038225",
    "https://eksisozluk.com/doktorlarin-artik-cerrahi-bransi-tercih-etmemeleri--5958479",
    "https://eksisozluk.com/dermatologlarin-hicbir-ise-yaramamasi--7763771",
    "https://eksisozluk.com/pediatri--145108",
    "https://eksisozluk.com/pediatri-doktoruna-ayar-veren-gece-bekcisi--8025398",
    "https://eksisozluk.com/kadin-dogum-doktoruna-verilen-77-milyonluk-ceza--8059241",
]

# Deduplicate while preserving order
_seen: set = set()
TARGET_THREADS = [u for u in TARGET_THREADS if not (u in _seen or _seen.add(u))]

BATCH_A = TARGET_THREADS[:15]   # test batch — run first, inspect before proceeding
BATCH_B = TARGET_THREADS[15:]   # full remaining batch


# ── LOGGING ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ── BROWSER SETUP ─────────────────────────────────────────────────────────────

def build_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Returns a Chrome driver configured to avoid bot detection.
    headless=True is only safe once the cookie/login flow has been verified
    visually — Eksi Sozluk blocks headless UA strings by default.
    """
    options = webdriver.ChromeOptions()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    # Mask the navigator.webdriver JS flag
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ── OVERLAY UTILITIES ─────────────────────────────────────────────────────────

def _js_click(driver: webdriver.Chrome, element) -> None:
    driver.execute_script("arguments[0].click();", element)


def dismiss_overlays(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    """Dismisses cookie banners and login popups using JS clicks."""
    time.sleep(1.5)

    consent_selectors = [
        "button.btn-cookie",
        "button[data-action='cookie-modal#accept']",
        "#cookie-modal button.btn-primary",
        "button.cookie-accept",
        "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'kabul')]",
        "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept')]",
    ]
    for selector in consent_selectors:
        try:
            btn = (
                driver.find_element(By.XPATH, selector)
                if selector.startswith("//")
                else driver.find_element(By.CSS_SELECTOR, selector)
            )
            _js_click(driver, btn)
            logger.info("Cookie banner dismissed.")
            time.sleep(0.8)
            break
        except NoSuchElementException:
            continue
        except Exception as e:
            logger.debug(f"Cookie dismiss attempt failed ({selector}): {e}")

    login_close_selectors = [
        "button.close-button",
        "a.close",
        "#login-modal button.close",
        "button[aria-label='close']",
        "//button[contains(@class,'close')]",
    ]
    for selector in login_close_selectors:
        try:
            btn = (
                driver.find_element(By.XPATH, selector)
                if selector.startswith("//")
                else driver.find_element(By.CSS_SELECTOR, selector)
            )
            _js_click(driver, btn)
            logger.info("Login overlay dismissed.")
            time.sleep(0.8)
            break
        except NoSuchElementException:
            continue


# ── DATE PARSER ───────────────────────────────────────────────────────────────

def parse_entry_date(raw_date: str) -> str:
    """
    Normalises Eksi Sozluk date strings to 'DD.MM.YYYY HH:MM'.
    Discards edit timestamps (right side of ~).
    """
    if not raw_date:
        return ""
    return raw_date.split("~")[0].strip()


# ── PAGE SCRAPER ──────────────────────────────────────────────────────────────

def scrape_page(driver: webdriver.Chrome, url: str, thread_slug: str) -> list[dict]:
    """
    Scrapes a single page of an Eksi Sozluk thread.
    Uses '#entry-item-list li[data-id]' to select only top-level entry elements
    (the data-id attribute is absent on nested li elements inside entry text).
    """
    entries = []
    driver.get(url)
    time.sleep(random.uniform(2.5, 4.5))

    try:
        wait = WebDriverWait(driver, 20)
        dismiss_overlays(driver, wait)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#entry-item-list")))

        entry_elements = driver.find_elements(By.CSS_SELECTOR, "#entry-item-list li[data-id]")
        if not entry_elements:
            logger.warning(f"No entries found on page: {url}")
            return entries

        for item in entry_elements:
            try:
                content_text = item.find_element(By.CSS_SELECTOR, "div.content").text.strip()

                try:
                    raw_date = item.find_element(By.CSS_SELECTOR, "a.entry-date").text.strip()
                except NoSuchElementException:
                    raw_date = ""

                try:
                    author_el = item.find_element(By.CSS_SELECTOR, "a[title][href*='/biri/']")
                    author = author_el.get_attribute("title") or author_el.text.strip()
                except NoSuchElementException:
                    author = ""

                entry_id = item.get_attribute("data-id") or ""

                entries.append({
                    "entry_id":    entry_id,
                    "date_raw":    raw_date,
                    "date_parsed": parse_entry_date(raw_date),
                    "author":      author,
                    "content":     content_text,
                    "thread":      thread_slug,
                    "url":         url,
                })

            except NoSuchElementException as e:
                logger.warning(f"Skipping malformed entry in {thread_slug}: {e}")

    except TimeoutException:
        logger.error(f"Timeout waiting for entry list on: {url}")

    return entries


# ── THREAD SCRAPER ────────────────────────────────────────────────────────────

def scrape_thread(
    driver: webdriver.Chrome,
    thread_url: str,
    max_pages: int = 10,
) -> pd.DataFrame:
    """
    Scrapes multiple pages of a single Eksi Sozluk thread.
    Stops early on the first empty page (past the last page).
    """
    thread_slug = thread_url.rstrip("/").split("/")[-1]
    all_entries: list[dict] = []

    for page_num in range(1, max_pages + 1):
        page_url = f"{thread_url}?p={page_num}"
        logger.info(f"  Scraping {thread_slug} — page {page_num}")

        page_entries = scrape_page(driver, page_url, thread_slug)
        if not page_entries:
            logger.info(f"  No entries on page {page_num}, stopping thread early.")
            break

        all_entries.extend(page_entries)

        if page_num < max_pages:
            sleep_time = random.uniform(4, 8)
            logger.info(f"  Sleeping {sleep_time:.1f}s before next page...")
            time.sleep(sleep_time)

    logger.info(f"  -> {len(all_entries)} total entries scraped from {thread_slug}")
    return pd.DataFrame(all_entries)


# ── BATCH RUNNER ──────────────────────────────────────────────────────────────

def run_batch(
    thread_urls: list[str],
    max_pages: int,
    output_file: Path,
) -> None:
    """
    Scrapes a list of threads and appends results to output_file after each
    thread, so progress is not lost if the process crashes mid-run.
    """
    driver = build_driver(headless=False)
    failed: list[str] = []
    first_write = True

    for i, url in enumerate(thread_urls, start=1):
        logger.info(f"[{i}/{len(thread_urls)}] {url}")
        try:
            df = scrape_thread(driver, url, max_pages=max_pages)
            if df.empty:
                logger.warning("  -> No data returned.")
                failed.append(url)
            else:
                df.to_csv(
                    output_file,
                    mode="a",
                    header=first_write,
                    index=False,
                    encoding="utf-8-sig",
                )
                first_write = False
                logger.info(f"  -> {len(df)} entries written to {output_file}")
        except Exception as e:
            logger.error(f"  -> Error: {e}")
            failed.append(url)

        sleep_time = random.uniform(6, 12)
        logger.info(f"  Sleeping {sleep_time:.1f}s...\n")
        time.sleep(sleep_time)

    driver.quit()

    if failed:
        logger.warning(f"\nFailed threads ({len(failed)}):")
        for t in failed:
            logger.warning(f"  {t}")
    else:
        logger.info("All threads scraped successfully.")


# ── MERGE & FILTER ────────────────────────────────────────────────────────────

def merge_and_filter() -> None:
    """
    Merges batch A and batch B CSVs, deduplicates by entry_id,
    filters to DATE_CUTOFF_YEAR–2025, saves raw and final corpora.
    """
    df_a = pd.read_csv(OUTPUT_BATCH_A, encoding="utf-8-sig")
    df_b = pd.read_csv(OUTPUT_BATCH_B, encoding="utf-8-sig")
    df = pd.concat([df_a, df_b], ignore_index=True)
    print(f"Raw entries loaded: {len(df)} (A: {len(df_a)}, B: {len(df_b)})")

    before = len(df)
    df = df.drop_duplicates(subset="entry_id")
    print(f"After dedup: {len(df)} (dropped {before - len(df)} duplicates)")

    df["year"] = pd.to_datetime(df["date_parsed"], dayfirst=True, errors="coerce").dt.year

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    print(f"Raw corpus saved: {OUTPUT_RAW}")

    df_filtered = df[(df["year"] >= DATE_CUTOFF_YEAR) & (df["year"] <= 2025)].copy()
    dropped = len(df) - len(df_filtered)
    print(f"Dropped {dropped} entries outside {DATE_CUTOFF_YEAR}–2025")

    df_filtered.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"Final corpus saved: {OUTPUT_FINAL} ({len(df_filtered)} entries)")

    print("\n--- Year distribution ---")
    print(df_filtered["year"].value_counts().sort_index().to_string())
    print(f"\nTotal threads : {df_filtered['thread'].nunique()}")
    print(f"Total entries : {len(df_filtered)}")
    print(f"Date range    : {int(df_filtered['year'].min())} – {int(df_filtered['year'].max())}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Total threads : {len(TARGET_THREADS)}")
    print(f"Batch A       : {len(BATCH_A)} threads (test, max 2 pages)")
    print(f"Batch B       : {len(BATCH_B)} threads (full, max 10 pages)")

    print("\n=== Batch A ===")
    run_batch(BATCH_A, max_pages=2, output_file=OUTPUT_BATCH_A)

    df_a = pd.read_csv(OUTPUT_BATCH_A, encoding="utf-8-sig")
    print(f"Batch A result: {len(df_a)} entries, {df_a['thread'].nunique()} threads")
    print(df_a[["thread", "date_parsed", "content"]].head(5).to_string(index=False))
    print("\nReview the output above. If it looks correct, batch B will now run.")

    print("\n=== Batch B ===")
    run_batch(BATCH_B, max_pages=10, output_file=OUTPUT_BATCH_B)

    print("\n=== Merge & Filter ===")
    merge_and_filter()


if __name__ == "__main__":
    main()
