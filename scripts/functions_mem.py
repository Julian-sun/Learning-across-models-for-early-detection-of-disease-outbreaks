"""
functions_mem.py

Implementation of the Moving Epidemic Method (MEM) for influenza/ILI surveillance.

Author
------
Juliane Oliveira
Centre for Data and Knowledge Integration for Health (CIDACS/FIOCRUZ)

Date
----
January 2026

Description
-----------
This module provides helper functions to:
1. Estimate epidemic periods using the Moving Epidemic Method (MEM).
2. Compute baseline and epidemic thresholds.
3. Calculate thresholds for different epidemic intensity levels.

References
----------
- Vega, T. et al. (2015). Influenza surveillance in Europe: establishing epidemic
  thresholds by the Moving Epidemic Method. Influenza and Other Respiratory Viruses, 9(5), 234–246.
"""


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import gmean
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------------------------------------------------------
# Utility: epidemiological season based on a cutoff week.
# -------------------------------------------------------------------
def add_sea(data, year_col="epiyear", week_col="epiweek", n_week=36):
    """
    Add epidemiological season column based on a cutoff week.

    Season definition:
    - weeks >= n_week belong to the next year's season
    - weeks < n_week belong to the current year's season
    The season is labeled by the ending year.
    """

    col = f"season_w{n_week}"

    data[col] = data[year_col]
    data.loc[data[week_col] >= n_week, col] += 1

    return data

# -------------------------------------------------------------------
# Utility: Check validity conditions for selected years.
# -------------------------------------------------------------------
def summary_is_valid(
    summary,
    years_to_check,
    k_start_min=10, # substituir por 30 // len(years_to_check) | quando len(years_to_check) >= 3 | = 10
    r_max=15,
    year_col="epiyear",
):
    """
    Check validity conditions only for selected years.
    """

    sub = summary[summary[year_col].isin(years_to_check)]

    # if no rows for those years, treat as invalid (safer)
    if sub.empty:
        return False

    return (
        (sub["k_start"] > k_start_min).all()
        and
        (sub["r_j_estr"] <= r_max).all()
    )

# -------------------------------------------------------------------
# Utility: Find best season definition and delta for MEM
# -------------------------------------------------------------------
def find_best_season_and_delta(
    df,
    city,
    season_weeks=[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 0],
    delta_start=0.02,
    delta_max=0.04,
    delta_step=0.001,
    k_start_min=10,
    r_max=15,
):
    """
    Identify the first combination of seasonal definition and MEM delta
    that produces a valid epidemic period summary for a given city.

    The function iterates over different season definitions (based on
    alternative starting weeks) and delta values used in the MEM
    epidemic period detection. For each combination, it computes the
    epidemic summary and checks whether it satisfies predefined
    validation criteria.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing epidemiological time series data for
        multiple municipalities. Must include at least:
        - `co_ibge` : municipality identifier
        - `atend_ivas` : raw time series of ILI attendances
        - `co_uf`, `nm_uf` : state information

    city : int or str
        IBGE code identifying the municipality to be analyzed.

    season_weeks : tuple of int, optional (default=(42, 32, 0))
        Candidate starting weeks used to define alternative epidemiological
        seasons. Each value is passed to `add_sea` to create a season/year
        column.

    delta_start : float, optional (default=0.02)
        Minimum delta value to be tested in the MEM algorithm.

    delta_max : float, optional (default=0.04)
        Maximum delta value to be tested (exclusive).

    delta_step : float, optional (default=0.001)
        Step size used to generate the sequence of delta values.

    k_start_min : int, optional (default=10)
        Minimum allowed value for the epidemic start week (`k_start`)
        used in the summary validation step.

    r_max : int, optional (default=20)
        Maximum allowed epidemic duration (`r`) used in the summary
        validation step.

    Returns
    -------
    pandas.DataFrame or None
        A copy of the first epidemic summary that satisfies the validation
        criteria, augmented with metadata:
        - `co_ibge` : municipality code
        - `co_uf`, `nm_uf` : state identifiers
        - `season_def` : season definition used
        - `delta_used` : delta value used in MEM

        Returns `None` if no valid combination of season definition and
        delta is found.

    Notes
    -----
    - The ILI time series is smoothed using a 4-week moving average
      before applying the MEM algorithm.
    - The function stops at the *first* valid combination found; it does
      not evaluate all possible combinations exhaustively.
    - This function relies on the external functions `add_sea`,
      `mem_epidemic_period`, and `summary_is_valid`.
    """

    # subset city
    set_muni = (
        df[df.co_ibge == city]
        .copy()
        .reset_index(drop=True)
    )

    # smooth series
    set_muni["atend_ivas_ma"] = (
        set_muni["atend_ivas"]
        .rolling(window=4, min_periods=1)
        .mean()
    )

    deltas = np.arange(delta_start, delta_max- 1e-9, +delta_step)

    for w in season_weeks:
        # create season definition
        set_muni = add_sea(set_muni, n_week=w)
        season_col = f"season_w{w}"

        for delta in deltas:
            
            summary, _ = mem_epidemic_period(
                set_muni,
                col_year=season_col,
                col_series="atend_ivas_ma",
                delta=delta,
            )


            if summary_is_valid(summary, years_to_check=[2023, 2024], k_start_min=k_start_min, r_max=r_max):
                out = summary.copy()
                out["co_ibge"] = city
                out["season_def"] = season_col
                out["delta_used"] = delta
                out["co_uf"] = set_muni.co_uf.iloc[0]
                out["nm_uf"] = set_muni.nm_uf.iloc[0]
                return out

    return None


# -------------------------------------------------------------------
# Utility: LOWESS smoothing
# -------------------------------------------------------------------
def smooth_lowess(pr, frac=0.3):
    """
    Apply LOWESS smoothing to the MAP curve.

    Parameters
    ----------
    pr : array-like
        MAP percentages for each window length.
    frac : float, optional
        Fraction of data used for smoothing (default=0.3).

    Returns
    -------
    np.ndarray
        Smoothed MAP curve.
    """
    r = np.arange(1, len(pr) + 1)
    smoothed = lowess(pr, r, frac=frac, return_sorted=False)
    return smoothed


# -------------------------------------------------------------------
# Step 1: Epidemic period estimation
# -------------------------------------------------------------------
def mem_epidemic_period(set_muni, col_year = 'epiyear', col_series = 'atend_ivas_ma',  delta=0.02):
    """
    Compute epidemic period (step 1 of MEM) for each season/year.

    Parameters
    ----------
    set_muni : DataFrame
        Must contain columns ['epiyear', 'atend_ivas'].
    delta : float, optional
        Threshold for slope increment (default=0.02).

    Returns
    -------
    summary : DataFrame
        Columns: ['epiyear', 'r_j_estr', 'k_start', 'k_end']
    details : dict
        Keys = epiyear, values = dict with:
            - r_j_estr : estimated epidemic duration (weeks)
            - k_start  : start week (1-indexed)
            - k_end    : end week (inclusive, 1-indexed)
            - p_j_r    : MAP percentages
            - sm_p_j_r : smoothed MAP curve
            - t_j_r    : cumulative totals by window length
    """
    
    results = {}
    summary_records = []

    for year in sorted(set_muni[col_year].unique()):
        s1 = set_muni[set_muni[col_year] == year].reset_index(drop=True)

        t = np.asarray(s1[col_series])
        S = len(t)
        tS = np.sum(t)

        # --- compute t_j_r (max accumulated per window length) ---
        t_j_r = np.array([
            sliding_window_view(t, r).sum(axis=1).max()
            for r in range(1, S + 1)
        ])

        # --- MAP curve ---
        p_j_r = t_j_r / tS
        sm_p_j_r = smooth_lowess(p_j_r)

        # --- increments Δr ---
        delta_j_r = np.diff(sm_p_j_r)

        # --- determine optimal duration r_j_estr ---
        indices = np.where(delta_j_r < delta)[0]
        r_j_estr = int(indices.min()) if len(indices) > 0 else S

        # --- find k_estr (start index of epidemic window) ---
        window_r = sliding_window_view(t, r_j_estr)
        window_sums = window_r.sum(axis=1)
        k_estr = window_sums.argmax()

        k_start = k_estr + 1          # 1-indexed
        k_end   = k_estr + r_j_estr   # inclusive

        # --- save results ---
        results[year] = {
            'r_j_estr': r_j_estr,
            'k_start': k_start,
            'k_end': k_end,
            'p_j_r': p_j_r,
            'sm_p_j_r': sm_p_j_r,
            't_j_r': t_j_r
        }

        summary_records.append({
            'epiyear': year,
            'r_j_estr': r_j_estr,
            'k_start': k_start,
            'k_end': k_end
        })

    summary = pd.DataFrame(summary_records)

    return summary, results

# -------------------------------------------------------------------
# Step 2: Baselines and thresholds
# -------------------------------------------------------------------
def baseline_thresholds(
    dta,
    summary_df,
    lst_sea,
    value_col="atend_ivas",
    col_year="epiyear",
    col_week="epiweek"
):
    """
    Compute baselines and epidemic thresholds from seasonal surveillance data.
    
    Parameters
    ----------
    dta : pd.DataFrame
        Full dataset containing at least columns: ["epiyear", "epiweek", value_col].
    lst_sea : list
        List of epidemic years to include in the calculation.
    value_col : str
        Column name with weekly counts (default = 'atend_ivas').
    
    Returns
    -------
        - baseline
        - post_baseline
        - epidemic_threshold
        - post_threshold
        - df_thresholds_intensity (DataFrame with 50%, 90%, 95% thresholds)
    """
    
    pre_values, post_values, epi_values = [], [], []
    pre_top, post_top, epi_top = [], [], []

    n = max(1, round(30 / len(lst_sea)))

    for year in lst_sea:
        season = dta[dta[col_year] == year]
        if season.empty:
            continue
        
        k_s = int(summary_df.loc[summary_df['epiyear'] == year, "k_start"].iloc[0])
        k_e = int(summary_df.loc[summary_df['epiyear'] == year, "k_end"].iloc[0])

        season["id"] = range(1, len(season) + 1)

        pre = season.loc[season['id'] <= k_s, value_col].values
        post = season.loc[season['id'] > k_e, value_col].values
        epi = season.loc[
            (season['id'] > k_s) & (season['id'] <= k_e),
            value_col,
        ].values

        pre_values.extend(pre)
        post_values.extend(post)
        epi_values.extend(epi)

        pre_top.extend(sorted(pre, reverse=True)[:n])
        post_top.extend(sorted(post, reverse=True)[:n])
        epi_top.extend(sorted(epi, reverse=True)[:n])

    pre_values, post_values, epi_values = map(np.array, [pre_values, post_values, epi_values])
    pre_top, post_top, epi_top = map(np.array, [pre_top, post_top, epi_top])

    baseline = np.mean(pre_values)
    post_baseline = np.mean(post_values)

    z = norm.ppf(0.95)

    epidemic_threshold = np.mean(pre_top) + z * np.std(pre_top, ddof=1)
    post_threshold = np.mean(post_top) + z * np.std(post_top, ddof=1)

    epi_top = epi_top[epi_top > 0]
    log_vals = np.log(epi_top)

    thresholds_intensity = {
        f"{int(p*100)}%": np.exp(np.mean(log_vals) + norm.ppf(p) * np.std(log_vals, ddof=1))
        for p in [0.50, 0.90, 0.95]
    }

    df_thresholds_intensity = pd.DataFrame(
        thresholds_intensity.items(), columns=["percentile", "value"]
    )

    return  baseline,  post_baseline, epidemic_threshold, post_threshold, df_thresholds_intensity 
    
