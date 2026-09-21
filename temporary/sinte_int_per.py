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

from pathlib import Path


# Read data with model outputs
#df = pd.read_parquet('/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/LR_MV_xgb_mlp_sintetic_08_05_2026.parquet')

#df_meta = pd.read_parquet('/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/sintetic_outbreak_metadata_20260325.parquet')

df = pd.read_parquet('/home/juliane.oliveira/workspace/Data/LR_MV_xgb_mlp_sintetic_08_05_2026.parquet')
#'/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/LR_MV_xgb_mlp_sintetic_08_05_2026.parquet')

df_meta = pd.read_parquet('/home/juliane.oliveira/workspace/Data/sintetic_outbreak_metadata_20260325.parquet')

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

# Classify MEM surges accordingly to intensity
df_mem = pd.read_parquet('/home/juliane.oliveira/workspace/Data/mem_output_26_03_2026.parquet')
#/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/mem_output_26_03_2026.parquet')

df_mem = df_mem[df_mem.epiyear.isin([2022,2023, 2024,2025])]

dta_intens = df_mem.groupby(['co_ibge'])[['baseline', 'post_baseline', 'epidemic_threshold',
       'post_threshold', 'low_level', 'medium_level', 'high_level']].max().reset_index()

# =============================================================================
# FUNCTION TO ASSIGN INTENSITY
# =============================================================================

def classify_block(values, base, epi, low, med, high):

    # Check from most severe to least

    if (values > high).any():
        return 'very high'

    elif (values > med).any():
        return 'high'

    elif (values > low).any():
        return 'medium'

    elif (values > epi).any():
        return 'low'

    elif ((values > base) & (values <= epi)).any():
        return 'very low'

    elif (values <= base).all():
        return 'baseline'

    return np.nan


# =============================================================================
# CREATE INTENSITY LABELS
# =============================================================================

def ints_creator(df, dta_intens):

    lst = []

    # Automatically detect replicate columns
    rep_cols = [
        c for c in df.columns
        if c.startswith('replicate_') and c.endswith('_surge_consec')
    ]

    for code, set_muni in df.groupby('co_ibge'):

        print(f'Processing municipality: {code}')

        set_muni = (
            set_muni
            .copy()
            .sort_values('year_week')
        )

        # Municipality-specific thresholds
        set_muni_inten = dta_intens[
            dta_intens.co_ibge == code
        ]

        # Skip if thresholds do not exist
        if set_muni_inten.empty:

            #print(f'No thresholds found for {code}')
            lst.append(set_muni)
            continue

        thresholds = set_muni_inten.iloc[0]

        # Thresholds
        base = int(thresholds['baseline'])
        epi  = int(thresholds['epidemic_threshold'])

        low = (
            int(thresholds['low_level'])
            if pd.notna(thresholds.get('low_level'))
            else 2 * epi
        )

        med = (
            int(thresholds['medium_level'])
            if pd.notna(thresholds.get('medium_level'))
            else 4 * epi
        )

        high = (
            int(thresholds['high_level'])
            if pd.notna(thresholds.get('high_level'))
            else 6 * epi
        )

        print(
            f'base={base}, epi={epi}, low={low}, med={med}, high={high}'
        )

        # ---------------------------------------------------------------------
        # Process each replicate column
        # ---------------------------------------------------------------------

        for col in rep_cols:

            #print(f'Processing {col}')

            # Extract replicate name
            rep_id = col.replace('_surge_consec', '')

            # Binary signal
            flag = set_muni[col]

            # Create block IDs for consecutive sequences
            block = (flag != flag.shift()).cumsum()

            intensity_col = f'intensity_{rep_id}'

            set_muni[intensity_col] = np.nan

            # -----------------------------------------------------------------
            # Process only positive blocks
            # -----------------------------------------------------------------

            for block_id, group in set_muni.groupby(block):

                # Skip non-surge blocks
                if group[col].iloc[0] != 1:
                    continue

                values = group['atend_ivas']

                # Assign intensity
                intensity_label = classify_block(
                    values,
                    base,
                    epi,
                    low,
                    med,
                    high
                )

                set_muni.loc[
                    group.index,
                    intensity_col
                ] = intensity_label

        lst.append(set_muni)

    return pd.concat(lst, ignore_index=True)


import warnings
warnings.filterwarnings('ignore')

res_int = ints_creator(data, dta_intens)

# detect replicate IDs from intensity columns
rep_ids = [
    col.replace('intensity_replicate_', '')
    for col in res_int.columns
    if col.startswith('intensity_replicate_')
]

for rep in rep_ids:

    intensity_col = f'intensity_replicate_{rep}'
    surge_col = f'replicate_{rep}_surge'
    consec_col = f'replicate_{rep}_surge_consec'

    # --- NON-consecutive ---
    res_int[f'mem_sub_rep_{rep}'] = (
        ((res_int[intensity_col] == 'very low') | (res_int[intensity_col] == 'baseline')) &
        (res_int[surge_col] == 1)
    ).astype(int)
    
    res_int[f'mem_low_rep_{rep}'] = (
        (res_int[intensity_col] == 'low') &
        (res_int[surge_col] == 1)
    ).astype(int)

    res_int[f'mem_medium_rep_{rep}'] = (
        (res_int[intensity_col] == 'medium') &
        (res_int[surge_col] == 1)
    ).astype(int)

    res_int[f'mem_high_rep_{rep}'] = (
        (res_int[intensity_col] == 'high') &
        (res_int[surge_col] == 1)
    ).astype(int)

    res_int[f'mem_very_high_rep_{rep}'] = (
        (res_int[intensity_col] == 'very high')
        & (res_int[surge_col] == 1)
    ).astype(int)

    # --- CONSECUTIVE ---
    res_int[f'mem_sub_consec_rep_{rep}'] = (
        ((res_int[intensity_col] == 'very low') | (res_int[intensity_col] == 'baseline')) &
        (res_int[consec_col] == 1)
    ).astype(int)
    
    res_int[f'mem_low_consec_rep_{rep}'] = (
        (res_int[intensity_col] == 'low') &
        (res_int[consec_col] == 1)
    ).astype(int)

    res_int[f'mem_medium_consec_rep_{rep}'] = (
        (res_int[intensity_col] == 'medium') &
        (res_int[consec_col] == 1)
    ).astype(int)

    res_int[f'mem_high_consec_rep_{rep}'] = (
        (res_int[intensity_col] == 'high') &
        (res_int[consec_col] == 1)
    ).astype(int)

    res_int[f'mem_very_high_consec_rep_{rep}'] = (
        (res_int[intensity_col] == 'very high') &
        (res_int[consec_col] == 1)
    ).astype(int)


# =============================================================================
# FUNCTION TO EVALUATE A MODEL
# =============================================================================

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


metrics_keep = ['Sensitivity ',
                'Specificity',
                'PPV', 
                'NPV'
                ]

def evaluate_model(df, model_col, warning_pairs):

    perf_list = []
    timing_list = []

    for warn_col, warn_consec_col in warning_pairs:

        print(f'  Evaluating {warn_col}')

        # ---------------------------------------------------------
        # Compute anticipation metrics
        # ---------------------------------------------------------

        df_warning_count = des_fun.antici_count(
            df,
            model_col,
            warn_col,
            warn_consec_col,
            'co_ibge'
        )

        performance_summary = des_fun.summarize_performance(
            df_warning_count
        )

        # ---------------------------------------------------------
        # Add metadata
        # ---------------------------------------------------------

        performance_summary['model'] = model_col
        performance_summary['intensity_level'] = warn_col

        
        performance_summary = performance_summary[
                                performance_summary.Metric.isin(metrics_keep)
                                ].copy()

        performance_summary['Value'] = (
            performance_summary['Value']
            .str.replace('%', '', regex=False)
            .astype(float)
            / 100
            )

        performance_summary['model'] = performance_summary['model'].replace(
            replace_dict,
            regex=True
                )

        df_warning_count['model'] = model_col
        df_warning_count['intensity_level'] = warn_col

        perf_list.append(performance_summary)
        timing_list.append(df_warning_count)

    # -------------------------------------------------------------
    # Return combined dataframes
    # -------------------------------------------------------------

    perf_df = pd.concat(perf_list, ignore_index=True)

    timing_df = pd.concat(timing_list, ignore_index=True)

    return perf_df, timing_df



# -------------------------------------------------------------
# Output directory
# -------------------------------------------------------------

out_dir = Path( "/home/juliane.oliveira/workspace/Data/sint_out_timing_per"
   # "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/sint_out_timing_per"
)

out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Main loop
# -------------------------------------------------------------

for rep in range(32):

    print(f'\n===== Replicate {rep} =====')

    levels = [
        'sub',
        'low',
        'medium',
        'high',
        'very_high'
    ]

    warning_pairs = [
        (
            f'mem_{level}_rep_{rep}',
            f'mem_{level}_consec_rep_{rep}'
        )
        for level in levels
    ]

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

    perf_rep_list = []
    timing_rep_list = []

    # ---------------------------------------------------------
    # Run models
    # ---------------------------------------------------------

    for model in models:

        print(f'Running model: {model}')

        perf_df, timing_df = evaluate_model(
            res_int,
            model,
            warning_pairs
        )

        perf_df['rep'] = rep
        timing_df['rep'] = rep

        perf_rep_list.append(perf_df)
        timing_rep_list.append(timing_df)

    # ---------------------------------------------------------
    # Save replicate results
    # ---------------------------------------------------------

    perf_rep = pd.concat(perf_rep_list, ignore_index=True)

    timing_rep = pd.concat(timing_rep_list, ignore_index=True)

    perf_rep.to_parquet(
        out_dir / f'performance_rep_{rep}.parquet',
        index=False
    )

    timing_rep.to_parquet(
        out_dir / f'timing_rep_{rep}.parquet',
        index=False
    )

    print(f'Saved replicate {rep}')