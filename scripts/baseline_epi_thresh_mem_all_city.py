#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Juliane Oliveira
Affiliation: CIDACS / FIOCRUZ

AESOP – Moving Epidemic Method (MEM)
Step 2: Compute baselines and epidemic thresholds per municipality

This script:
- loads municipal-level surveillance data,
- loads the selected seasonal definition and delta per municipality,
- applies the MEM to estimate epidemic periods,
- computes baselines and intensity thresholds,
- and saves the final MEM outputs.

"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

import functions_mem as fm


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling"
INPUT_DATA_FILE = "aesop_2026_01_21_mun.parquet"
SEASON_KEY_FILE = max(Path(DATA_PATH).glob("def_sea_MEM_out_*.parquet" ), key=lambda x: x.stat().st_mtime)

OUTPUT_PREFIX = "mem_output"

# Seasons used to estimate baselines and thresholds
SEASONS_FOR_BASELINE = [2022, 2023, 2024]

# Columns required for MEM processing
COLUMNS_TO_KEEP = [
    "co_uf",
    "nm_uf",
    "nm_municipio",
    "co_ibge",
    "epiyear",
    "epiweek",
    "year_week",
    "atend_totais",
    "atend_ivas",
    "ra_atend_ivas",
    "ra_atend_ivas_ma",
]


# ============================================================
# Load data
# ============================================================

df = pd.read_parquet(Path(DATA_PATH) / INPUT_DATA_FILE)
sea_key = pd.read_parquet(Path(DATA_PATH) / SEASON_KEY_FILE)

df = df[COLUMNS_TO_KEEP]

df = df[df.year_week >= "2022-42"]


# ============================================================
# Run MEM: baselines and thresholds per municipality
# ============================================================

results = []

for co_ibge in df.co_ibge.unique():

    # --------------------------------------------------------
    # Retrieve selected season definition and delta
    # --------------------------------------------------------
    row = sea_key.loc[sea_key.co_ibge == co_ibge].head(1)

    if row.empty:
        print(f"⚠️ No season/delta found for municipality {co_ibge}")
        continue

    # Extract week number from strings like "season_w42", "season_w32", "season_w0"
    week_start_seas = int(
        row["season_def"]
        .str.extract(r"w(\d+)")
        .iloc[0, 0]
    )

    delta_used = row["delta_used"].iloc[0]

    # --------------------------------------------------------
    # Prepare municipal time series
    # --------------------------------------------------------
    set_muni = df[df.co_ibge == co_ibge].copy()

    # Smooth ILI/IVAS series using 4-week moving average
    set_muni["atend_ivas_ma"] = (
        set_muni["atend_ivas"]
        .rolling(window=4, min_periods=1)
        .mean()
    )

    # --------------------------------------------------------
    # Define season variable
    # --------------------------------------------------------
    # If week_start_seas == 0, use calendar epidemiological year
    if week_start_seas == 0:
        col_year = "epiyear"
    else:
        set_muni = fm.add_sea(set_muni, n_week=week_start_seas)
        col_year = f"season_w{week_start_seas}"

    # --------------------------------------------------------
    # Run MEM to identify epidemic periods
    # --------------------------------------------------------
    summary, details = fm.mem_epidemic_period(
        set_muni,
        col_year=col_year,
        col_series="atend_ivas_ma",
        delta=delta_used,
    )

    # --------------------------------------------------------
    # Compute baselines and intensity thresholds
    # --------------------------------------------------------
    (
        baseline,
        post_baseline,
        epidemic_threshold,
        post_threshold,
        df_thresholds_intensity,
    ) = fm.baseline_thresholds(
        set_muni,
        summary,
        lst_sea=SEASONS_FOR_BASELINE,
        value_col="atend_ivas",
        col_year=col_year,
        col_week="epiweek",
    )

    # --------------------------------------------------------
    # Order baselines and intensity thresholds
    # --------------------------------------------------------

    low_level=df_thresholds_intensity.value.iloc[0]
    medium_level=df_thresholds_intensity.value.iloc[1]
    high_level=df_thresholds_intensity.value.iloc[2]

    baseline, epidemic_threshold, low_level, medium_level, high_level = sorted([baseline, epidemic_threshold, low_level, medium_level, high_level])

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------
    summary = summary.assign(
        co_ibge=co_ibge,
        week_start_seas=week_start_seas,
        col_year_str=col_year,
        baseline=baseline,
        post_baseline=post_baseline,
        epidemic_threshold=epidemic_threshold,
        post_threshold=post_threshold,
        low_level=low_level, #df_thresholds_intensity.value.iloc[0],
        medium_level=medium_level,#df_thresholds_intensity.value.iloc[1],
        high_level = high_level #df_thresholds_intensity.value.iloc[2],
    )

    results.append(summary)


# ============================================================
# Save output
# ============================================================

final = pd.concat(results, ignore_index=True)

out_file = (
    Path(DATA_PATH)
    / f"{OUTPUT_PREFIX}_{datetime.now():%d_%m_%Y}.parquet"
)

final.to_parquet(out_file)

print(f"Saved MEM output to: {out_file}")
print(f"Number of municipalities processed: {final.co_ibge.nunique()}")






