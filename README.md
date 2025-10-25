# Purpose
Enrich news article samples with NER, sentiment, attack detection, perspective labeling, event clustering, and best-effort coordinate inference. Produce auditable tabular outputs and a GeoPackage ready for ArcGIS Pro.

# Run pipeline (already installed venv)
# to see if venv is active, your terminal should look similar to this if it is:
(venv) PS C:\Users\jscso\Desktop\gdelt_api_project>

# Activate venv (if not active):
PowerShell: .\venv\Scripts\Activate.ps1

# If user needs permission for venv:
PowerShell: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Quick status check
Confirm core packages are present:
pip show pandas spacy vaderSentiment scikit-learn requests pyarrow

# Run pipeline:
python main.py

# All outputs are written to data/test_output:
-enriched_articles_with_clusters.parquet
-enriched_articles_with_clusters.gpkg (if geopandas installed and rows have lon/lat) Geocoding not implemented yet!
-raw_responses_from_samples.jsonl
-signals_by_date_country.csv
-diagnostics.json

# What main.py does
Reads samples from data/samples/articles_sample.json or data/samples/articles_sample.csv.
Extracts entities using spaCy and computes VADER sentiment for titles.
Detects attack-related tokens and counts them.
Normalizes source country names via pycountry and assigns a perspective label (allied, russian, other).
Clusters likely same‑event reporting by date + country + text similarity (TF‑IDF + DBSCAN).
Infers coordinates via GeoNames API when GEO_NAMES_USERNAME is configured; otherwise falls back to country centroids and domain-TLD heuristics.
Writes enriched table, clustering metadata, diagnostics, and optionally a GeoPackage.

# Input files: how to add data to process
Place input file(s) into data/samples.
Preferred names and order: articles_sample.json then articles_sample.csv.

Each row should include columns where available: title, url, domain, seendate, sourcecountry.

# To process new input:
Place file into data/samples and re-run python main.py..

# CSV data summary
A cluster summary CSV is produced by the helper script data/scripts/make_cluster_summary.py which writes:
data/test_output/cluster_summary.csv

# Fields included in cluster_summary.csv:
cluster_id, cluster_size, sample_date, country_alpha2, perspective, mean_tone_local, attack_count_total, top_domains, sample_titles_urls

# ArcGIS Pro integration steps (when lon/lat available) //Could work haven't fully tested yet as geocoding not implemented
If GeoPackage exists:
In ArcGIS Pro → Map tab → Add Data → select data/test_output/enriched_articles_with_clusters.gpkg → layer "articles".

# If only parquet/csv exists:
Convert to GeoPackage locally with geopandas (see conversion snippet below) or add as table and join to spatial features after geocoding.

# Conversion snippet to create GPKG (run in the venv; requires geopandas):
python    // I can script this if needed, just haven't found an actual use until we can import anything into a GIS map
import pandas as pd, geopandas as gpd
from shapely.geometry import Point
df = pd.read_parquet("data/test_output/enriched_articles_with_clusters.parquet")
df = df.dropna(subset=["lon","lat"])
gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon.astype(float), df.lat.astype(float))], crs="EPSG:4326")
gdf.to_file("data/test_output/enriched_articles_with_clusters.gpkg", layer="articles", driver="GPKG")
Geocoding notes
To enable GeoNames API geocoding, set GEO_NAMES_USERNAME in main.py and re-run main.py.. GeoNames increases coordinate coverage by resolving entity place names to lon/lat.    // Will implement GEO_NAMES_USERNAME later

# Troubleshooting checklist
ModuleNotFoundError: ensure venv is activated or run with .\venv\Scripts\python.exe -m pip install <package>.

# Recreate environment on another machine if needed:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# From previous testing:
local_summary.json
enriched_articles.parquet    // New file for that involves cluster data

# Final note
remove_results currently isn't implemented. If we want remote data then I can begin to implement that later but right now everything is local only.