#!/usr/bin/env python3
# main.py
"""
GDELT sample enrichment pipeline with event clustering and geocoding.
Outputs (data/test_output):
 - raw_responses_from_samples.jsonl
 - enriched_articles_with_clusters.parquet (CSV fallback)
 - enriched_articles_with_clusters.gpkg (if geopandas available)
 - signals_by_date_country.csv
 - diagnostics.json
Requirements (already in venv):
 pip install pandas pycountry spacy vaderSentiment scikit-learn requests pyarrow
 python -m spacy download en_core_web_sm
 pip install geopandas fiona shapely  # to create GeoPackage
 pip install sentence-transformers hdbscan  # for improved semantic clustering (not used by default)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pycountry
import spacy
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# clustering / text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False

# File config - edit as needed
DATA_SAMPLES = Path("data/samples")
OUTPUT_DIR = Path("data/test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_JSON = DATA_SAMPLES / "articles_sample.json"
SAMPLES_CSV = DATA_SAMPLES / "articles_sample.csv"

LIVE_MODE = False            # set True to run live GDELT queries (requires gdeltdoc)
NUM_RECORDS = 50
DAY_WINDOW = 1               # +/- days window for live queries
RATE_SLEEP = 1.0             # seconds between live API calls

# GEO coding config
GEO_NAMES_USERNAME = ""      # set to your GeoNames username to enable place lookups. Blank leads to use country-centroid fallback
GEONAMES_RATE_SLEEP = 1.0    # seconds between GeoNames requests to be polite

# Perspective groups (alpha-2 codes)
ALLIED = {"US", "GB", "PL", "DE", "FR"}
RUSSIAN = {"RU", "BY"}

# NER + sentiment initialization
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

# attack tokens
ATTACK_TOKENS = {
    "attack", "attacked", "attacks", "invasion", "invaded", "shelling",
    "strike", "struck", "bomb", "bombing", "missile", "killed", "fired",
    "clash", "clashes", "offensive", "retake", "recapture"
}

# small country centroid fallback (lon, lat) for common countries. Can expand or just use the api later on
COUNTRY_CENTROIDS = {
    "UA": (31.1656, 48.3794),  # Ukraine center (lon, lat)
    "RU": (105.3188, 61.5240), # Russia
    "US": (-98.583, 39.833333),
    "GB": (-2.0, 54.0),
    "PL": (19.1451, 51.9194),
    "DE": (10.4515, 51.1657),
    "FR": (2.2137, 46.2276),
    "BY": (27.9534, 53.7098),
}

# try to import gdeltdoc only when LIVE_MODE used
try:
    from gdeltdoc import GdeltDoc, Filters
except Exception:
    GdeltDoc = None
    Filters = None


def load_samples():
    if SAMPLES_JSON.exists():
        return pd.read_json(SAMPLES_JSON, orient="records")
    if SAMPLES_CSV.exists():
        return pd.read_csv(SAMPLES_CSV)
    return pd.DataFrame()

def save_jsonl_raw(records, filename):
    fp = OUTPUT_DIR / filename
    with fp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(fp)

def safe_write_parquet_or_csv(df, base_name):
    base = OUTPUT_DIR / base_name
    try:
        outp = base.with_suffix(".parquet")
        df.to_parquet(outp, index=False)
        return str(outp)
    except Exception:
        outp = base.with_suffix(".csv")
        df.to_csv(outp, index=False)
        return str(outp)

def parse_seendate(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ")
    except Exception:
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return None

def extract_entities(text):
    if not text:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def detect_attack_tokens(text):
    if not text or not isinstance(text, str):
        return False, 0
    words = [w.lower().strip(".,:;\"'()") for w in text.split()]
    found = [w for w in words if w in ATTACK_TOKENS]
    return (len(found) > 0), len(found)

def local_vader_score(text):
    if not text or not isinstance(text, str):
        return None
    return vader.polarity_scores(text)["compound"]

def normalize_country_name(name):
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    try:
        c = pycountry.countries.lookup(name)
        return {"name": c.name, "alpha2": c.alpha_2, "alpha3": getattr(c, "alpha_3", None)}
    except Exception:
        lc = name.lower()
        mapping = {
            "united states": "US",
            "usa": "US",
            "uk": "GB",
            "russia": "RU",
            "belarus": "BY",
            "united kingdom": "GB",
            "ukraine": "UA",
        }
        if lc in mapping:
            code = mapping[lc]
            c = pycountry.countries.get(alpha_2=code)
            if c:
                return {"name": c.name, "alpha2": c.alpha_2, "alpha3": getattr(c, "alpha_3", None)}
    return None

def pick_primary_location_from_entities(entities):
    if not entities:
        return None
    for e in entities:
        if isinstance(e, dict) and e.get("label") in ("GPE", "LOC"):
            return e.get("text")
    return None

# GeoNames lookup. requires GEO_NAMES_USERNAME to be configured
def geonames_lookup_place(q, username):
    if not username:
        return None
    try:
        url = "http://api.geonames.org/searchJSON"
        params = {"q": q, "maxRows": 1, "username": username}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        js = r.json()
        if js.get("totalResultsCount", 0) > 0:
            rec = js["geonames"][0]
            lon = float(rec.get("lng"))
            lat = float(rec.get("lat"))
            cc = rec.get("countryCode")
            return {"lon": lon, "lat": lat, "country_code": cc, "name": rec.get("name")}
    except Exception:
        return None
    return None

def infer_coords_for_row(row, geonames_username=""):
    """
    Attempt to infer lon/lat for the article row.
    1) try first GPE/LOC entity via GeoNames (if username provided)
    2) else fallback to sourcecountry_norm centroid mapping
    Returns dict with lon, lat, method
    """
    ents = row.get("entities") or []
    candidate = pick_primary_location_from_entities(ents)
    if candidate and geonames_username:
        res = geonames_lookup_place(candidate, geonames_username)
        time.sleep(GEONAMES_RATE_SLEEP)
        if res:
            return {"lon": res["lon"], "lat": res["lat"], "method": "geonames_entity", "place_name": res.get("name"), "geocountry": res.get("country_code")}
    # Try domain TLD or sourcecountry_norm for alpha2
    sc = row.get("sourcecountry_norm")
    if isinstance(sc, dict):
        code = sc.get("alpha2")
        if code and code in COUNTRY_CENTROIDS:
            lon, lat = COUNTRY_CENTROIDS[code]
            return {"lon": lon, "lat": lat, "method": "country_centroid", "geocountry": code}
    # fallback to try entity without geonames or any numeric in text
    return {"lon": None, "lat": None, "method": None, "geocountry": None}

# Enrichment and signals

def enrich_dataframe(df):
    rows = []
    for _, r in df.iterrows():
        title = r.get("title", "") or ""
        url = r.get("url", "") or ""
        domain = r.get("domain") or ""
        sourcecountry = r.get("sourcecountry") or r.get("country") or None
        seendate_raw = r.get("seendate")
        seendate_dt = parse_seendate(seendate_raw)
        entities = extract_entities(title)
        attack_flag, attack_count = detect_attack_tokens(title)
        tone_local = local_vader_score(title)
        country_norm = normalize_country_name(sourcecountry)
        primary_loc = pick_primary_location_from_entities(entities)
        row_out = {
            "url": url,
            "title": title,
            "domain": domain,
            "sourcecountry_raw": sourcecountry,
            "sourcecountry_norm": country_norm,
            "seendate_raw": seendate_raw,
            "seendate": seendate_dt.isoformat() if seendate_dt is not None else None,
            "entities": entities,
            "primary_location": primary_loc,
            "attack_flag": bool(attack_flag),
            "attack_count": int(attack_count),
            "tone_local": float(tone_local) if tone_local is not None else None
        }
        # keep original columns if present
        for c in df.columns:
            if c not in row_out:
                row_out[c] = r.get(c)
        rows.append(row_out)
    enriched = pd.DataFrame(rows)
    return enriched

def generate_signals_by_date_country(enriched_df):
    df = enriched_df.copy()
    df["seendate_dt"] = pd.to_datetime(df["seendate"], errors="coerce")
    df["date"] = df["seendate_dt"].dt.date
    def pick_country_alpha2(x):
        if isinstance(x, dict):
            return x.get("alpha2")
        return None
    df["country_alpha2"] = df["sourcecountry_norm"].apply(pick_country_alpha2)
    grp = df.groupby("country_alpha2")
    rows = []
    for country, g in grp:
        country_key = country if pd.notna(country) else "UNKNOWN"
        row = {
            "country_alpha2": country_key,
            "articles_count": int(len(g)),
            "mean_tone_local": float(g["tone_local"].dropna().mean()) if not g["tone_local"].dropna().empty else None,
            "attack_count_total": int(g["attack_count"].sum()),
        }
        rows.append(row)
    signals = pd.DataFrame(rows)
    return signals

# clustering for same-event grouping (TF-IDF + DBSCAN)
def cluster_same_events(enriched_df, time_window_days=1, min_samples=1, eps=0.30, max_features=1000):
    df = enriched_df.copy()
    df["seendate_dt"] = pd.to_datetime(df["seendate"], errors="coerce")
    df["date"] = df["seendate_dt"].dt.date
    def pick_country_alpha2(x):
        if isinstance(x, dict):
            return x.get("alpha2")
        return None
    df["country_alpha2"] = df["sourcecountry_norm"].apply(pick_country_alpha2).fillna("UNKNOWN")
    df["cluster_id"] = None

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)

    next_cluster_global = 0
    grouped = df.groupby(["country_alpha2", "date"])
    for (country, date), group in grouped:
        if group.shape[0] == 0:
            continue
        titles = group["title"].fillna("").astype(str).tolist()
        if len(titles) == 1:
            idx0 = group.index[0]
            df.at[idx0, "cluster_id"] = f"{country}_{date}_c{next_cluster_global}"
            next_cluster_global += 1
            continue
        X = vectorizer.fit_transform(titles)
        sim = cosine_similarity(X)
        dist = 1.0 - sim
        db = DBSCAN(eps=eps, min_samples=max(1, min_samples), metric="precomputed")
        labels = db.fit_predict(dist)
        for local_idx, label in enumerate(labels):
            global_idx = group.index[local_idx]
            if label == -1:
                cid = f"{country}_{date}_c{next_cluster_global}"
                next_cluster_global += 1
            else:
                cid = f"{country}_{date}_c{label}"
            df.at[global_idx, "cluster_id"] = cid

    df["cluster_size"] = df.groupby("cluster_id")["cluster_id"].transform("size").fillna(1).astype(int)
    df["cluster_id"] = df["cluster_id"].astype(str)
    return df

# perspective assignment
def add_perspective_column(df):
    def pick_persp(x):
        if isinstance(x, dict):
            code = x.get("alpha2")
            if code in ALLIED:
                return "allied"
            if code in RUSSIAN:
                return "russian"
        return "other"
    df["perspective"] = df["sourcecountry_norm"].apply(pick_persp)
    return df

# coordinate inference for all rows
def infer_coordinates_for_df(df, geonames_username=""):
    rows = []
    for _, r in df.iterrows():
        coords = infer_coords_for_row(r, geonames_username=geonames_username)
        r_out = r.copy()
        r_out["lon"] = coords.get("lon")
        r_out["lat"] = coords.get("lat")
        r_out["geo_method"] = coords.get("method")
        r_out["geocountry"] = coords.get("geocountry")
        r_out["place_name_geocoded"] = coords.get("place_name")
        rows.append(r_out)
    out_df = pd.DataFrame(rows)
    return out_df

# Live test runner
def build_filters_from_row_for_live(row, day_window=0, keyword_seed=None):
    sd = row.get("seendate")
    dt = parse_seendate(sd)
    if dt is None:
        dt = datetime.utcnow()
    start_dt = datetime.combine(dt.date(), datetime.min.time()) - timedelta(days=day_window)
    end_dt = datetime.combine(dt.date(), datetime.max.time()).replace(microsecond=0) + timedelta(days=day_window)
    title = row.get("title") or ""
    title_low = title.lower()
    if keyword_seed:
        kw = keyword_seed
    else:
        if ("russia" in title_low) or ("ukraine" in title_low):
            kw = title
        else:
            kw = "russia ukraine invasion attack shelling missile"
    if Filters is None:
        return None
    f = Filters(start_date=start_dt, end_date=end_dt, keyword=kw, num_records=NUM_RECORDS)
    return f

def run_local_pipeline(geonames_username=""):
    df = load_samples()
    if df.empty:
        print("No sample files found in data/samples. Put sample JSON/CSV there and re-run.")
        return

    # Persist raw samples
    raw_records = df.to_dict(orient="records")
    raw_path = save_jsonl_raw(raw_records, "raw_responses_from_samples.jsonl")
    print("Saved raw_responses JSONL:", raw_path)

    # Enrich
    enriched = enrich_dataframe(df)

    # Cluster same-event articles
    enriched = cluster_same_events(enriched, time_window_days=DAY_WINDOW, min_samples=1, eps=0.30, max_features=1000)

    # Add perspective
    enriched = add_perspective_column(enriched)

    # Infer coordinates (GeoNames if username provided, otherwise fallback)
    enriched = infer_coordinates_for_df(enriched, geonames_username=geonames_username)

    # Persist enriched with clusters
    enriched_path = safe_write_parquet_or_csv(enriched, "enriched_articles_with_clusters")
    print("Saved enriched articles with clusters:", enriched_path)

    # try to write GeoPackage if geopandas available and lon/lat present
    try:
        if GEOPANDAS_AVAILABLE and not enriched[["lon", "lat"]].isnull().all().any():
            geo_df = enriched.dropna(subset=["lon", "lat"]).copy()
            geo_df["lon"] = geo_df["lon"].astype(float)
            geo_df["lat"] = geo_df["lat"].astype(float)
            geo_df["geometry"] = [Point(xy) for xy in zip(geo_df.lon, geo_df.lat)]
            gdf = gpd.GeoDataFrame(geo_df, geometry="geometry", crs="EPSG:4326")
            gpkg_path = OUTPUT_DIR / "enriched_articles_with_clusters.gpkg"
            gdf.to_file(gpkg_path, layer="articles", driver="GPKG")
            print("Wrote GeoPackage:", str(gpkg_path))
    except Exception as e:
        print("Could not write GeoPackage (geopandas error):", e)

    # Signals and diagnostics
    signals = generate_signals_by_date_country(enriched)
    signals_path = OUTPUT_DIR / "signals_by_date_country.csv"
    signals.to_csv(signals_path, index=False)
    print("Saved signals CSV:", signals_path)

    diag = {
        "sample_rows": int(len(df)),
        "enriched_rows": int(len(enriched)),
        "signals_rows": int(len(signals)),
        "top_domains": enriched["domain"].value_counts().head(10).to_dict(),
        "clusters_summary": enriched.groupby("cluster_size").size().to_dict(),
        "geocoded_count": int(enriched[enriched["lon"].notna()].shape[0]),
    }
    with (OUTPUT_DIR / "diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, ensure_ascii=False)
    print("Wrote diagnostics.json")

def run_live_tests():
    if GdeltDoc is None or Filters is None:
        print("gdeltdoc not available. Install gdeltdoc to run live queries.")
        return
    df_samples = load_samples()
    if df_samples.empty:
        print("No sample rows to base live queries on.")
        return
    client = GdeltDoc()
    raw_responses = []
    saved_frames = []
    for idx, row in df_samples.iterrows():
        print(f"Live query for sample {idx}: {row.get('title')}")
        f = build_filters_from_row_for_live(row, day_window=DAY_WINDOW)
        if f is None:
            print("Could not build Filters for row:", idx)
            continue
        if hasattr(f, "query_string"):
            print("DEBUG: Filters.query_string ->", f.query_string)
        try:
            df_res = client.article_search(f)
            df_res["_sample_idx"] = int(idx)
            raw_responses.append({"sample_idx": int(idx), "filters_query": getattr(f, "query_string", None), "rows": len(df_res)})
            saved_frames.append(df_res)
            time.sleep(RATE_SLEEP)
        except Exception as e:
            print("Live query failed:", e)
            raw_responses.append({"sample_idx": int(idx), "filters_query": getattr(f, "query_string", None), "error": str(e)})
            time.sleep(RATE_SLEEP)
    save_jsonl_raw(raw_responses, "raw_responses_live_meta.jsonl")
    if saved_frames:
        big = pd.concat(saved_frames, ignore_index=True)
        path = safe_write_parquet_or_csv(big, "live_articles")
        print("Saved live_articles to", path)
    else:
        print("No live frames returned")

def main():
    print("Running local enrichment pipeline...")
    # set GEO_NAMES_USERNAME to enable geocoding from GeoNames, else country centroids will be used
    run_local_pipeline(geonames_username=GEO_NAMES_USERNAME)
    if LIVE_MODE:
        print("\nRunning live queries (LIVE_MODE enabled)...")
        run_live_tests()
    print("\nDone. Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
