#!/usr/bin/env python3
# data/scripts/make_cluster_summary.py
import pandas as pd
from pathlib import Path
import json
# This file creates the summary. For now its missing geocoding to be able to integrate into GIS maps

IN_PQ = Path("data/test_output/enriched_articles_with_clusters.parquet")
OUT_CSV = Path("data/test_output/cluster_summary.csv")
DIAG = Path("data/test_output/diagnostics.json")

if not IN_PQ.exists():
    raise SystemExit(f"Input file not found: {IN_PQ}")

df = pd.read_parquet(IN_PQ)

# normalize cluster_id and fill missing cluster_id
df["cluster_id"] = df["cluster_id"].fillna("UNKNOWN_CLUSTER")
# parse seendate to datetime for aggregation
df["seendate_dt"] = pd.to_datetime(df["seendate"], errors="coerce")

def top_domains_semicolon(series, n=3):
    vc = series.fillna("").value_counts()
    items = [f"{d}" for d in vc.index.tolist()[:n]]
    return ";".join(items)

def sample_titles_urls(series_titles, series_urls, n=5):
    rows = []
    for t,u in zip(series_titles.fillna(""), series_urls.fillna("")):
        rows.append(f"{t.strip()} | {u.strip()}")
        if len(rows) >= n:
            break
    return ";".join(rows)

rows = []
for cid, group in df.groupby("cluster_id"):
    size = int(len(group))
    earliest = group["seendate_dt"].min()
    # majority country (alpha2) from normalized field if present
    def pick_country_alpha2(x):
        if isinstance(x, dict):
            return x.get("alpha2")
        return None
    group["country_alpha2"] = group["sourcecountry_norm"].apply(pick_country_alpha2)
    country_mode = group["country_alpha2"].dropna().mode()
    country_mode = country_mode.iloc[0] if not country_mode.empty else None
    # majority perspective
    persp_mode = group["perspective"].dropna().mode()
    persp_mode = persp_mode.iloc[0] if not persp_mode.empty else "other"
    mean_tone = group["tone_local"].dropna().mean()
    attack_total = int(group["attack_count"].sum()) if "attack_count" in group.columns else int(group["attack_count"].sum())
    top_domains = top_domains_semicolon(group["domain"], n=3)
    sample_summ = sample_titles_urls(group["title"], group["url"], n=5)
    rows.append({
        "cluster_id": cid,
        "cluster_size": size,
        "sample_date": earliest.date().isoformat() if pd.notna(earliest) else "",
        "country_alpha2": country_mode or "",
        "perspective": persp_mode,
        "mean_tone_local": round(float(mean_tone),4) if pd.notna(mean_tone) else "",
        "attack_count_total": attack_total,
        "top_domains": top_domains,
        "sample_titles_urls": sample_summ
    })

out_df = pd.DataFrame(rows).sort_values(["cluster_size","sample_date"], ascending=[False,True])
out_df.to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV)

# also echo diagnostics if available
if DIAG.exists():
    print("Diagnostics:")
    print(json.dumps(json.load(open(DIAG,"r",encoding="utf-8")), indent=2))
