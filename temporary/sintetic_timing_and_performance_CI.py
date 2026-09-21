#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

#from matplotlib import pyplot as plt

#import seaborn as sns

#from matplotlib_venn import venn3

import statsmodels.api as sm

#import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import descri_function as des_fun


# In[2]:


# Read data with model outputs
df = pd.read_parquet('/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/LR_MV_xgb_mlp_sintetic_08_05_2026.parquet')

df_meta = pd.read_parquet('/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/sintetic_outbreak_metadata_20260325.parquet')


# # Create the columns indicating the onset and periods for the sintetic outbreak inserted

# convert to weekly period (ISO-like)
df = df.assign(yw = pd.to_datetime(df['year_week'] + '-1', format='%Y-%W-%w').dt.to_period('W'))
df_meta = df_meta.assign(start_yw = pd.to_datetime(df_meta['start_year_week'] + '-1', format='%Y-%W-%w').dt.to_period('W'))
df_meta = df_meta.assign(end_yw   = pd.to_datetime(df_meta['end_year_week'] + '-1', format='%Y-%W-%w').dt.to_period('W'))



def compute_replicates(df, df_meta):
    df = df.copy()

    for code, set_muni in df.groupby('co_ibge'):
        meta_city = df_meta[df_meta['co_ibge'] == code]

        warning = set_muni['warning_final_mem_surge_01'].values
        yw = set_muni['yw'].values

        for rep, dta in meta_city.groupby('replicate'):

            # --- STEP 1: identify NON-coincident starts ---
            is_start = set_muni['yw'].isin(dta['start_yw']).values
            noncoincident_start = is_start & (warning == 0)

            # store start signal
            df.loc[set_muni.index, f'replicate_{rep}_surge'] = noncoincident_start.astype(int)

            # --- STEP 2: keep only intervals whose start is non-coincident ---
            valid_intervals = dta[dta['start_yw'].isin(set_muni.loc[noncoincident_start, 'yw'])]

            if len(valid_intervals) == 0:
                df.loc[set_muni.index, f'replicate_{rep}_surge_consec'] = 0
                continue

            # build consecutive mask WITHOUT warning condition
            cond_consec = (
                (yw[:, None] >= valid_intervals['start_yw'].values) &
                (yw[:, None] <= valid_intervals['end_yw'].values)
            ).any(axis=1)

            df.loc[set_muni.index, f'replicate_{rep}_surge_consec'] = cond_consec.astype(int)

    return df


result = compute_replicates(df, df_meta)



data = result[result.year_week >= '2024-51']


# =============================================================================
# MODEL COLLECTION
# =============================================================================

models = {
    'EARS': data.filter(regex=r'^C2_alarms_'),
    'EVI': data.filter(regex=r'^sinal_evi_replicate'),
    'ISF': data.filter(regex=r'^EWS_ISF_replicate'),
    'LOF': data.filter(regex=r'^EWS_LOF_replicate'),
    'OCSVM': data.filter(regex=r'^EWS_OCSVM_replicate'),
    'COPOD': data.filter(regex=r'^EWS_COPOD_replicate'),
    'NGM': data.filter(regex=r'^EWS_Rt_replicate'),
    'MV': data.filter(regex=r'^hard_voting_ivas_rep'),
    'LR': data.filter(regex=r'^sinal_ens_ivas_rep'),
    'XGB': data.filter(regex=r'^signal_ensemble_xgb50_rep'),
    'MLP': data.filter(regex=r'^signal_ensemble_mlp50')
}



# # Prepare data


lst_timing = []
lst_metrics_res = []

# ---------------------------------------------------
# Model name cleaner
# ---------------------------------------------------

replace_dict = {
    r'^EWS_COPOD_replicate_\d+$': 'COPOD',
    r'^EWS_ISF_replicate_\d+$': 'ISF',
    r'^EWS_LOF_replicate_\d+$': 'LOF',
    r'^EWS_OCSVM_replicate_\d+$': 'OCSVM',
    r'^EWS_Rt_replicate_\d+$': 'NGM',
    r'^C2_alarms_\d+$': 'EARS',
    r'^sinal_evi_replicate_\d+$': 'EVI',
    r'^signal_ensemble_xgb50_rep_\d+$': 'XGBOOST',
    r'^signal_ensemble_mlp50_rep_\d+$': 'MLP',
    r'^sinal_ens_ivas_rep_\d+$': 'LR',
    r'^hard_voting_ivas_rep_\d+$': 'MV'
}

metrics_keep = [
    'Sensitivity ',
    'Specificity',
    'PPV',
    'NPV'
]

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------

for rep in range(32):

    print(rep)

    models = [
        f'C2_alarms_{rep}',
        f'sinal_evi_replicate_{rep}',
        f'EWS_ISF_replicate_{rep}',
        f'EWS_LOF_replicate_{rep}',
        f'EWS_OCSVM_replicate_{rep}',
        f'EWS_COPOD_replicate_{rep}',
        f'EWS_Rt_replicate_{rep}',
        f'hard_voting_ivas_rep_{rep}',
        f'sinal_ens_ivas_rep_{rep}',
        f'signal_ensemble_xgb50_rep_{rep}',
        f'signal_ensemble_mlp50_rep_{rep}'
    ]

    warn_col = f'replicate_{rep}_surge'
    warn_col_with_consec = f'replicate_{rep}_surge_consec'

    lst1 = []
    lst2 = []

    for sel_model in models:

        print(sel_model)

        df_warning_count = des_fun.antici_count(
            data,
            sel_model,
            warn_col,
            warn_col_with_consec,
            'co_ibge'
        )

        performance_summary = des_fun.summarize_performance(
            df_warning_count
        )

        performance_summary['model'] = sel_model
        df_warning_count['model'] = sel_model

        lst1.append(performance_summary)
        lst2.append(df_warning_count)

    # ---------------------------------------------------
    # Combine results
    # ---------------------------------------------------

    summary = pd.concat(lst1, ignore_index=True)

    dta_tab1 = pd.concat(lst2, ignore_index=True)

    # Clean model names
    dta_tab1['model'] = dta_tab1['model'].replace(
        replace_dict,
        regex=True
    )

    # Timing summary
    dta_tab2 = (
        dta_tab1
        .groupby('model')[
            ['n3', 'n2', 'n1', 'n0', 'missed', 'total_aih_warning']
        ]
        .sum()
        .reset_index()
    )

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    dta3 = summary[
        summary.Metric.isin(metrics_keep)
    ].copy()

    dta3['Value'] = (
        dta3['Value']
        .str.replace('%', '', regex=False)
        .astype(float)
        / 100
    )

    dta3['model'] = dta3['model'].replace(
        replace_dict,
        regex=True
    )

    # Add replicate info
    dta_tab2['rep'] = rep
    dta3['rep'] = rep

    lst_timing.append(dta_tab2)
    lst_metrics_res.append(dta3)



dta_tim = pd.concat(lst_timing)
dta_metrics = pd.concat(lst_metrics_res)


from pathlib import Path
from datetime import datetime

# =============================================================================
# CONCATENATE RESULTS
# =============================================================================

dta_tim = pd.concat(lst_timing, ignore_index=True)
dta_metrics = pd.concat(lst_metrics_res, ignore_index=True)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

out_dir = Path(
    "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling"
)

# Create directory if it does not exist
out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FILE NAMES
# =============================================================================

today = datetime.now().strftime("%d_%m_%Y")

fname_tim = f"timing_results_{today}.parquet"
fname_metrics = f"metrics_results_{today}.parquet"

# =============================================================================
# SAVE PARQUET FILES
# =============================================================================

dta_tim.to_parquet(
    out_dir / fname_tim,
    index=False
)

dta_metrics.to_parquet(
    out_dir / fname_metrics,
    index=False
)

print(f"Saved timing results to: {out_dir / fname_tim}")
print(f"Saved metrics results to: {out_dir / fname_metrics}")





