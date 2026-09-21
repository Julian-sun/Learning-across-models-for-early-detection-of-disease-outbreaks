#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author
------
Juliane Oliveira
Centre for Data and Knowledge Integration for Health (CIDACS/FIOCRUZ)

Aim
---
Run Moving Epidemic Method (MEM) to select the best seasonal definition
and delta parameter for each municipality.

This script is part of the AESOP project.
"""

# =========================
# Imports
# =========================

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Scientific libraries
from scipy.stats import norm
from scipy.stats.mstats import gmean
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# I/O
import pyarrow.parquet as pq

# Local functions
import functions_mem as fm


# =========================
# Configuration
# =========================

# Input data
DATA_PATH = Path(
    "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/"
    "aesop_2026_01_21_mun.parquet"
)

# Output directory
OUT_DIR = Path(
    "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling"
)

# Columns used in the analysis
COLS = [
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


# =========================
# Load data
# =========================

df = pd.read_parquet(DATA_PATH)
df = df[COLS]


# =========================
# Run MEM for all cities
# =========================

lst=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 0]

results = []

for city in df.co_ibge.unique():
    print(f"Processing municipality {city}...")
    res = fm.find_best_season_and_delta(df, city, season_weeks = lst)
    if res is not None:
        results.append(res)

# Concatenate successful results
final_selected = pd.concat(results, ignore_index=True)


# =========================
# Save outputs
# =========================

# 1. Selected season and delta per municipality
fname_out = f"def_sea_MEM_out_{datetime.now():%d_%m_%Y}.parquet"
final_selected.to_parquet(OUT_DIR / fname_out)

# 2. Municipalities without valid MEM configuration
bad_cities = set(df.co_ibge.unique()) - set(final_selected.co_ibge.unique())
df_bad = pd.DataFrame({"co_ibge": sorted(bad_cities)})

fname_bad = f"set_bad_cities_{datetime.now():%d_%m_%Y}.csv"
df_bad.to_csv(OUT_DIR / fname_bad, index=False)

print(f"Finished. Valid cities: {final_selected.co_ibge.nunique()}")

