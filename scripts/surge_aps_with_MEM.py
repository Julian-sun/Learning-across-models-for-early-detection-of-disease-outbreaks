#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AESOP – Moving Epidemic Method (MEM)
Step 3: Define epidemic surges and generate warning signals

This script:
- loads MEM thresholds estimated per municipality,
- applies MEM-based surge definitions to weekly IVAS counts,
- cleans isolated warnings and enforces temporal consistency,
- generates fallback surge definitions for municipalities without MEM results,
- and saves the final warning dataset for visualization and analysis.

Author: Juliane Oliveira
Affiliation: CIDACS / FIOCRUZ
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
import warnings

from datetime import datetime
from pathlib import Path

import functions_mem as fm
import functions  # auxiliary warning-cleaning utilities

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling"

MEM_OUTPUT_FILE = max(Path(DATA_PATH).glob("mem_output_*parquet"), key=lambda x: x.stat().st_mtime)
RAW_DATA_FILE = "aesop_2026_01_21_mun.parquet"

OUTPUT_PREFIX = "aesop_with_MEM"

# ============================================================
# Load data
# ============================================================

# MEM thresholds and parameters per municipality
df_mem = pd.read_parquet(Path(DATA_PATH) / MEM_OUTPUT_FILE)

# Original municipal surveillance data
df_raw = pd.read_parquet(Path(DATA_PATH) / RAW_DATA_FILE)

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
]

df_raw = df_raw[COLUMNS_TO_KEEP]


# Index MEM for performance
df_mem_indexed = df_mem.set_index("co_ibge")


# ============================================================
# Apply MEM-based surge definitions
# ============================================================

results_mem = []

for co_ibge in df_mem.co_ibge.unique():
    print(f"Processing municipality {co_ibge}...")

    try:
        set_muni = df_raw[df_raw.co_ibge == co_ibge].copy()

        # Retrieve MEM thresholds
        #baseline = df_mem.loc[df_mem.co_ibge == co_ibge, "baseline"].max()
        epidemic_threshold = df_mem.loc[df_mem.co_ibge == co_ibge, "epidemic_threshold"].max()

        #low_level = df_mem.loc[df_mem.co_ibge == co_ibge, "low_level"].max()
        #medium_level = df_mem.loc[df_mem.co_ibge == co_ibge, "medium_level"].max()
        #high_level = df_mem.loc[df_mem.co_ibge == co_ibge, "high_level"].max()

        # ----------------------------------------------------
        # Define surge thresholds
        # ----------------------------------------------------
        threshold_base = epidemic_threshold #max(baseline, low_level)

        # Binary MEM surge (0/1)
        set_muni["mem_surge_01"] = pd.cut(
                set_muni["atend_ivas"],
                bins=[-np.inf, threshold_base, np.inf],#bins=[0, threshold_base, set_muni["atend_ivas"].max() + 1],
                labels=[0, 1],
                include_lowest=True,
            ).astype(int)

       
        # Multi-level MEM surge (0–3)
        #set_muni["mem_surge"] = pd.cut(
        #        set_muni["atend_ivas"],
        #        bins=[
        #            0,
        #            threshold_base,
        #            low_level,
        #            medium_level,
        #            high_level,
        #            high_level + set_muni["atend_ivas"].max() + 1,
        #        ],
        #        labels=[0, 1, 2, 3,4],
        #        include_lowest=True,
        #    ).astype(int)

        # Prefer MEM intensity thresholds when available
        #if pd.notna(low_level):

        #    threshold_base = epidemic_threshold #max(baseline, low_level)

        #    # Binary MEM surge (0/1)
        #    set_muni["mem_surge_01"] = pd.cut(
        #        set_muni["atend_ivas"],
        #        bins=[0, threshold_base, set_muni["atend_ivas"].max() + 1],
        #        labels=[0, 1],
        #        include_lowest=True,
        #    ).astype(int)

        #    # Multi-level MEM surge (0–3)
        #    set_muni["mem_surge"] = pd.cut(
        #        set_muni["atend_ivas"],
        #        bins=[
        #            0,
        #            threshold_base,
        #            low_level,
        #            medium_level,
        #            high_level,
        #            high_level + set_muni["atend_ivas"].max() + 1,
        #        ],
        #        labels=[0, 1, 2, 3,4],
        #        include_lowest=True,
        #    ).astype(int)

        # ----------------------------------------------------
        # Fallback: use epidemic threshold only
        # ----------------------------------------------------
        #else:
        #    threshold_base = epidemic_threshold
        #
        #    set_muni["mem_surge_01"] = pd.cut(
        #        set_muni["atend_ivas"],
        #        bins=[0, threshold_base, set_muni["atend_ivas"].max() + 1],
        #        labels=[0, 1],
        #        include_lowest=True,
        #    ).astype(int)

        #    set_muni["mem_surge"] = set_muni["mem_surge_01"]

        results_mem.append(set_muni)

    except Exception as e:
        print(f"⚠️ Skipping {co_ibge} due to error: {e}")
        continue


df_mem_surge = pd.concat(results_mem, ignore_index=True)


# ============================================================
# Clean isolated warnings and enforce temporal consistency
# ============================================================

df_mem_cleaned = functions.clean_warning_column(
    df_mem_surge,
    group_col="co_ibge",
    time_col="year_week",
    warning_col="mem_surge_01",
)

df_mem_cleaned = df_mem_cleaned.rename(
    columns={
        "cleaned_warning": "mem_surge_01_without_isolated",
        "event": "mem_surge_01_correct_with_consec",
        "warning_final": "warning_final_mem_surge_01",
    }
)


# ============================================================
# Alternative surge definition (municipalities without MEM)
# ============================================================

missing_mem = set(df_raw.co_ibge.unique()) - set(df_mem_cleaned.co_ibge.unique())

fallback_results = []

for co_ibge in missing_mem:

    set_muni = df_raw[df_raw.co_ibge == co_ibge].copy()

    # Simple statistical baseline
    mean_val = set_muni.atend_ivas.mean()
    median_val = set_muni.atend_ivas.median()
    std_val = set_muni.atend_ivas.std()

    set_muni["surge_ivas_alternative"] = (
        set_muni["atend_ivas"] > median_val + 2 * std_val
    ).astype(int)

    fallback_results.append(set_muni)


df_fallback = pd.concat(fallback_results, ignore_index=True)

df_fallback_cleaned = functions.clean_warning_column(
    df_fallback,
    group_col="co_ibge",
    time_col="year_week",
    warning_col="surge_ivas_alternative",
)

df_fallback_cleaned = df_fallback_cleaned.rename(
    columns={
        "cleaned_warning": "mem_surge_01_without_isolated",
        "event": "mem_surge_01_correct_with_consec",
        "warning_final": "warning_final_mem_surge_01",
    }
)


# ============================================================
# Merge MEM-based and fallback results
# ============================================================

final_mem = df_mem_cleaned[
    [
        "co_uf",
        "nm_uf",
        "nm_municipio",
        "co_ibge",
        "epiyear",
        "epiweek",
        "year_week",
        "atend_totais",
        "atend_ivas",
        'mem_surge_01_correct_with_consec',
        "warning_final_mem_surge_01",
    ]
]

final_fallback = df_fallback_cleaned[
    [
        "co_uf",
        "nm_uf",
        "nm_municipio",
        "co_ibge",
        "epiyear",
        "epiweek",
        "year_week",
        "atend_totais",
        "atend_ivas",
        'mem_surge_01_correct_with_consec',
        "warning_final_mem_surge_01",
    ]
]

final = pd.concat([final_mem, final_fallback], ignore_index=True)


# ============================================================
# Save output
# ============================================================

out_file = (
    Path(DATA_PATH)
    / f"{OUTPUT_PREFIX}_{datetime.now():%d_%m_%Y}.parquet"
)

final.to_parquet(out_file)

print(f"Saved final MEM warning dataset to: {out_file}")

