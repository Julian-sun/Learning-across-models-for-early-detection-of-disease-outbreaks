#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import as_completed, ProcessPoolExecutor
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
import math
# import seaborn as sns
# import pyarrow as pa
# import pyarrow.parquet as pq
import itertools
import evi_functions as evi_func
import warnings
from datetime import datetime
from pathlib import Path


# Read data
ensamble = Path("/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/")
df = pd.read_parquet(ensamble / "aesop_with_MEM_26_03_2026.parquet")

df = df[df.year_week >= '2022-42']


# Run code for all cities

warnings.filterwarnings("ignore")

out_dir = Path("/opt/storage/raw/aesop/visualization/ensamble_modelling/evi_par")
#fname = f"evi_best_par_muni_{datetime.now():%d_%m_%Y}.parquet"
#out_path = out_dir / fname

out_dir.mkdir(exist_ok=True, parents=True)


def find_params_evi(df: pd.DataFrame, code: int) -> None:
    # set_muni = df[df.co_ibge == code].copy()

    dtf = evi_func.func(5, df.atend_ivas.to_numpy(), 2, 8, 0.2)

    df = df.reset_index(drop=True).copy()
    df["evi_t1_t"] = dtf["evi_t1_t"]
    df["ind"] = dtf["ind"]


    max_value = np.nanmax(df.loc[np.isfinite(df['evi_t1_t']), 'evi_t1_t'])
    df['evi_t1_t'].replace([np.inf, -np.inf], max_value, inplace=True)

    df["sinal_evi_ivas"] = df["ind"].astype(int)

    #if len(df_m) < 52:  # example: require ≥1 year
    #    continue

    m_values = np.arange(3, 20, 1)
    c_values = np.arange(0.1, int(df.evi_t1_t.max()) - 0.3, 0.1)

    df_res = evi_func.grid_search_m_c(m_values, c_values, df)
    #df_res = df_res[(df_res["sensitivity"] >= 0.5) & (df_res['specificity'] >= 0.5)]

    if len(df_res) == 0:
        ...
        # continue

    
    df_constrained = df_res[(df_res["sensitivity"] >= 0.5) & (df_res['specificity'] >= 0.5)]

    if df_constrained.empty:
        ...
        # continue  

    best = df_constrained.loc[df_constrained["youden_J"].idxmax()]
    best["co_ibge"] = code
       
    best_df = pd.DataFrame([best])
    print("salvando: ", Path(out_dir / f"best_{code}.parquet"))
    best_df.to_parquet(out_dir / f"best_{code}.parquet")


with ProcessPoolExecutor(max_workers=16) as pool:
    for code in df.co_ibge.unique():
        pool.submit(find_params_evi, df[df.co_ibge == code].copy(), code)
