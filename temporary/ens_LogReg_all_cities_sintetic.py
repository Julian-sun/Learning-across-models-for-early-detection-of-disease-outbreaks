
"""
===============================================================================
Ensemble Early Warning System (EWS) Modelling using Majority Voting and
Logistic Regression for Synthetic Replicates
===============================================================================

Author: Juliane Oliveira
Institution: Centre for Data and Knowledge Integration for Health (CIDACS),
             Gonçalo Moniz Institute, Oswaldo Cruz Foundation (FIOCRUZ)

Date: May 2026

Description
-----------
This script implements an ensemble modelling framework for infectious disease
early warning systems (EWS) using synthetic replicated datasets.

The workflow combines:
    1. Majority Voting (MV)
    2. Logistic Regression (LR)

across multiple anomaly detection and surveillance models.

The objective is to generate ensemble outbreak warning signals for each
municipality (`co_ibge`) and epidemiological week (`year_week`) using
different synthetic replicates.

The script:
    - Reads synthetic surveillance datasets stored in parquet format
    - Computes majority voting ensemble signals
    - Generates lagged features for temporal modelling
    - Fits municipality-level logistic regression models
    - Produces probabilistic outbreak risk estimates
    - Saves the final ensemble results as a parquet dataset

-------------------------------------------------------------------------------
Required Packages
-------------------------------------------------------------------------------

Python >= 3.9

Required libraries:
    pandas
    numpy
    pyarrow
    statsmodels
    scikit-learn

Install with:

    pip install pandas numpy pyarrow matplotlib matplotlib-venn \
                 statsmodels scikit-learn

-------------------------------------------------------------------------------
Expected Input Data
-------------------------------------------------------------------------------

The input parquet dataset must contain at least the following columns:

Identifiers:
    - co_ibge
    - year_week

Target variables:
    - mem_surge_01_replicate_{rep}
    - mem_surge_01_replicate_{rep}_correct_with_consec

Model outputs for each replicate:
    - C2_alarms_{rep}
    - sinal_evi_replicate_{rep}
    - EWS_ISF_replicate_{rep}
    - EWS_LOF_replicate_{rep}
    - EWS_OCSVM_replicate_{rep}
    - EWS_COPOD_replicate_{rep}
    - EWS_Rt_replicate_{rep}

where {rep} corresponds to replicate numbers (e.g., 0–31).

-------------------------------------------------------------------------------
Outputs
-------------------------------------------------------------------------------

The script generates:
    - Majority voting ensemble signals
    - Logistic regression risk probabilities
    - Final ensemble outbreak warning signals

Saved as:
    LR_MV_sintetic_<date>.parquet

===============================================================================
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


import statsmodels.api as sm


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pathlib import Path
from datetime import datetime


# =============================================================================
# Load Input Dataset
# =============================================================================

df = pd.read_parquet(
    '/opt/storage/shared/aesop/aesop_shared/'
    'ensamble_modelling/output_eadms_sintetic_07_05_2026.parquet'
)


# =============================================================================
# Ensemble Modelling Function
# =============================================================================




def lr_mv(data, replicates=range(32), lags=[1, 2, 3]):
     """
    Perform ensemble modelling using Majority Voting and Logistic Regression.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing municipality-level surveillance data.

    replicates : iterable, optional
        Replicate identifiers to process.
        Default is range(32).

    lags : list, optional
        Temporal lags used to generate lagged predictors.
        Default is [1, 2, 3].

    Returns
    -------
    pandas.DataFrame
        Dataframe containing:
            - majority voting ensemble outputs
            - logistic regression probabilities
            - final ensemble warning signals

    Notes
    -----
    Workflow:
        1. Generate majority voting ensemble signals
        2. Create lagged temporal features
        3. Train municipality-level logistic regression models
        4. Generate probabilistic risk predictions
        5. Produce final binary ensemble warning signals
    """


    results = []

    for rep in replicates:

        print(f"\n===== Processing replicate {rep} =====")

        # -----------------------------
        # Dynamic column names
        # -----------------------------
        lst_models = [
            f'C2_alarms_{rep}',
            f'sinal_evi_replicate_{rep}',
            f'EWS_ISF_replicate_{rep}',
            f'EWS_LOF_replicate_{rep}',
            f'EWS_OCSVM_replicate_{rep}',
            f'EWS_COPOD_replicate_{rep}',
            f'EWS_Rt_replicate_{rep}'
        ]

        col_surge = f'mem_surge_01_replicate_{rep}'
        col_surge_consec = f'mem_surge_01_replicate_{rep}_correct_with_consec'

        # -----------------------------
        # Create lagged column names
        # -----------------------------
        lagged_cols = []

        for var in lst_models:
            for lag in lags:
                lagged_cols.append(f"{var}_lag_{lag}")

        all_features = lst_models + lagged_cols

        # -----------------------------
        # Ensure binary integer type
        # -----------------------------
        data[lst_models] = data[lst_models].astype(int)

        # -----------------------------
        # Majority voting
        # -----------------------------
        threshold = round(len(lst_models) / 2, 0)

        lst_cities = []

        for code in data.co_ibge.unique():

            print(f"Processing MV for {code}...")

            set_muni = (
                data[data.co_ibge == code]
                .sort_values(['year_week'])
                .copy()
            )

            set_muni[f'hard_voting_ivas_rep_{rep}'] = (
                set_muni[lst_models].sum(axis=1) >= threshold
            ).astype(int)

            lst_cities.append(set_muni)

        data_rep = pd.concat(lst_cities)

        # -----------------------------
        # Logistic regression
        # -----------------------------
        lst = []

        for code in data_rep.co_ibge.unique():

            print(f"Processing LR for {code}...")

            set_muni = (
                data_rep[data_rep.co_ibge == code]
                .sort_values(['year_week'])
                .copy()
            )

            # Create lagged features
            for var in lst_models:
                for lag in lags:

                    set_muni[f"{var}_lag_{lag}"] = (
                        set_muni[var]
                        .shift(lag)
                        .fillna(0)
                        .astype(int)
                    )

            # Train only if enough positives
            if set_muni[col_surge].sum() > 1 and set_muni[col_surge_consec].nunique() > 1:

                X = set_muni[all_features].fillna(0)
                y = set_muni[col_surge_consec]

                try:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=500,
                        stratify=y
                    )

                    clf = LogisticRegression(
                        class_weight="balanced",
                        max_iter=500
                    )

                    clf.fit(X_train, y_train)

                    set_muni[f"risk_probs_rep_{rep}"] = (
                        clf.predict_proba(X)[:, 1]
                    )

                    set_muni[f"sinal_ens_ivas_rep_{rep}"] = (
                        set_muni[f"risk_probs_rep_{rep}"] > 0.5
                    ).astype(int)

                except Exception as e:

                    print(f"Error for {code}, rep {rep}: {e}")

                    set_muni[f"sinal_ens_ivas_rep_{rep}"] = (
                        set_muni[f'hard_voting_ivas_rep_{rep}']
                    )

            else:

                set_muni[f"sinal_ens_ivas_rep_{rep}"] = (
                    set_muni[f'hard_voting_ivas_rep_{rep}']
                )

            lst.append(set_muni)

        rep_result = pd.concat(lst)

        # Keep only new columns to avoid duplication
        keep_cols = [
            'co_ibge',
            'year_week',
            f'hard_voting_ivas_rep_{rep}',
            f'risk_probs_rep_{rep}',
            f'sinal_ens_ivas_rep_{rep}'
        ]

        keep_cols = [c for c in keep_cols if c in rep_result.columns]

        results.append(rep_result[keep_cols])

    # ---------------------------------
    # Merge all replicate results
    # ---------------------------------
    final_res = data.copy()

    for df_rep in results:
        final_res = final_res.merge(
            df_rep,
            on=['co_ibge', 'year_week'],
            how='left'
        )

    return final_res


# =============================================================================
# Run Ensemble Model
# =============================================================================

# Example for selected replicates:
# final_df = lr_mv(df, replicates=[0, 1])

# Run all replicates
final_df = lr_mv(df)


# =============================================================================
# Save Output
# =============================================================================

out_dir = Path(
    "/opt/storage/shared/aesop/aesop_shared/ensamble_modelling"
)

fname = f"LR_MV_sintetic_{datetime.now():%d_%m_%Y}.parquet"

final_df.to_parquet(out_dir / fname)

print(f"\nOutput saved to: {out_dir / fname}")