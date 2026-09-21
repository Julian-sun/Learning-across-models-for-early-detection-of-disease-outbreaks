import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
import math


def func(m, series, r_a, mu, c):
    '''
    This function calculates the EVIt-1,t which can be further used for the creation of the EVI ind.
    The imput funcion is: 
    m: the rolling window size; 
    serie: the time series where we desire to employ our result;
    r_a: the lag for the mooving average to be applied to the series; 
    mu: the window for posterior mean (used in the original method as 7)
    c is a treshold limit
    '''

    tseries = pd.Series(series).rolling(window=r_a).mean()
    
    # Obtain the rolling windows of size m and calculate their standard deviation
    std_windows = []
    
    for i in range(0, len(tseries) - m + 1):
        win = tseries[0+i:m+i]
        std_windows.append(win.std())
    
    # Calculate EVIt-1,t

    evi_t1_t = round(
        (pd.Series(std_windows) - pd.Series(std_windows).shift()) /
        abs(pd.Series(std_windows).shift()), 2
    )

    evi_t1_t_2 = [0] * (m-1) + evi_t1_t.to_list()

    # Obtain the mean of the last 7 points

    mean_windows = []

    for i in range(0, len(tseries) - mu + 1):
        win = series[0+i:mu+i]

        mean_windows.append(win.mean())

    mu_ = [0] * (mu-1) + mean_windows

    # Create a dataframe with the results
    d = {
        'series': series,
        'tseries': tseries,
        'evi_t1_t': evi_t1_t_2,
        'mu': mu_
    }


    data = pd.DataFrame(data=d)

    data.evi_t1_t = data.evi_t1_t.fillna(0)

    data.mu = round(data.mu, 0)

    data.loc[(data['series'] < 0.75 * data['mu']), 'cat_mu'] = 0
    data.loc[((data['series'] >= 0.75 * data['mu']) &
              (data['series'] < data['mu'])), 'cat_mu'] = 0.5

    data.loc[(data['series'] >= data['mu']), 'cat_mu'] = 1

    data.loc[(data['evi_t1_t'] < c), 'cat_evi'] = 0
    # data.loc[(data['evi_t1_t'] >= c1) & (data['evi_t1_t'] < c2),'cat_evi'] = 0.5
    data.loc[(data['evi_t1_t'] >= c), 'cat_evi'] = 1

    data.loc[(data['cat_evi'] == 0), 'ind'] = 0
    data.loc[(data['cat_evi'] == 1) & (data['cat_mu'] == 0), 'ind'] = 0
    data.loc[(data['cat_evi'] == 1) & (data['cat_mu'] == 0.5), 'ind'] = 0
    # data.loc[(data['cat_evi'] == 0) & (data['cat_mu'] == 1),'ind'] = 0.5
    data.loc[(data['cat_evi'] == 1) & (data['cat_mu'] == 1), 'ind'] = 1

    data.loc[(data['cat_mu'] == 0) | (data['evi_t1_t'] < 0), 'ind_2'] = 0
    # data.loc[(data['evi_t1_t'] = 0) & (data['cat_mu'] == 1),'ind_2'] = 0
    data.loc[(data['evi_t1_t'] >= 0) &
             (data['evi_t1_t'] <= c) & (data['cat_mu'] >= 0.5), 'ind_2'] = 0.5

    data.loc[(data['evi_t1_t'] > c) & (data['cat_mu'] == 0.5), 'ind_2'] = 0.5
    data.loc[(data['evi_t1_t'] > c) & (data['cat_mu'] == 1), 'ind_2'] = 1

    return data
    

def plot_func(dtf, c):

    x = list(range(0, len(dtf)))

    dtf = dtf.assign(Days=x)

    fig = plt.figure(figsize=(15, 3))
    plt.plot(x, dtf.series)
    plt.plot(x, dtf.tseries, '-o')
    plt.plot(x, dtf.mu, linestyle='--', label='mu')
    # plt.plot(x, 0.75*dtf.mu,linestyle = '--',label = '0.75mu')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()

    # plot2

    fig = plt.figure(figsize=(15, 3))
    plt.plot(x, dtf.evi_t1_t)
    plt.axhline(y=c, color='y', linestyle='--', label='c')
    # plt.axhline(y = c2, color = 'r', linestyle = '--', label='c2')
    plt.legend()
    plt.ylim([-1, 1])
    plt.xlabel('Days')
    plt.ylabel('EVI_t-1,t')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # plot3
    fig = plt.figure(figsize=(15, 4))
    sns.scatterplot(x="Days", y="tseries", data=dtf, hue="ind",
                    palette="RdYlBu_r")  # gist_gray, flare,RdYlBu_r

    # plot4
    fig = plt.figure(figsize=(15, 4))
    sns.scatterplot(x="Days", y="tseries", data=dtf, hue="ind_2",
                    palette="RdYlBu_r")  # gist_gray, flare,RdYlBu_r
    plt.plot(x, dtf.mu, linestyle='--', label='mu')


def plot_func2(dtf, c):

    # x = list(range(0, len(dtf)))
    x = dtf.year_week_ts
    # dtf = dtf.assign(Days = x)

    fig = plt.figure(figsize=(15, 3))
    plt.plot(x, dtf.atend_ivas)
    plt.plot(x, dtf.tseries, '-o')
    plt.plot(x, dtf.mu, linestyle='--', label='mu')
    # plt.plot(x, 0.75*dtf.mu,linestyle = '--',label = '0.75mu')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()

    # plot2

    fig = plt.figure(figsize=(15, 3))
    plt.plot(x, dtf.evi_t1_t)
    plt.axhline(y=c, color='y', linestyle='--', label='c')
    # plt.axhline(y = c2, color = 'r', linestyle = '--', label='c2')
    plt.legend()
    plt.ylim([-1, 1])
    plt.xlabel('year_week_ts')
    plt.ylabel('EVI_t-1,t')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # plot3

    fig = plt.figure(figsize=(15, 4))
    sns.scatterplot(x="year_week_ts", y="tseries", data=dtf, hue="ind",
                    palette="RdYlBu_r")  # gist_gray, flare,RdYlBu_r
    plt.xticks(rotation=90)

    # plot4

    fig = plt.figure(figsize=(15, 4))
    sns.scatterplot(x="year_week_ts", y="tseries", data=dtf, hue="ind2",
                    palette="RdYlBu_r")  # gist_gray, flare,RdYlBu_r
    plt.xticks(rotation=90)


def antici_count(data_res, col_warn_s1, col_warn_s2,col_warn_s2_with_consec, col_code):
    """
    Function to compute anticipated counts of warnings and missed warnings 
    across different lead times per unique region.
    
    Parameters:
    data_res (pd.DataFrame): Input DataFrame.
    col_warn_s1 (str): Column name for the primary warning signal (e.g., PHC warnings).
    col_warn_s2 (str): Column name for the secondary warning signal (e.g., AIH warnings).
    col_code (str): Column name identifying the region.

    Returns:
    pd.DataFrame: Summary DataFrame with counts of early, concurrent, and missed warnings.
    """
    
    lst_count = []

    for code in data_res[col_code].unique():
    
        dta = data_res[data_res[col_code] == code].copy().reset_index()  # Use only data for the current region

        # encontra todos os surtos que estão sendo detectados 3 semanas antes
        set3 = dta[(dta[col_warn_s1] == 1) & (dta[col_warn_s2].shift(-3) == 1)].index + 3 
        # encontra todos os surtos que estão sendo detectados 2 semanas antes e excluindo aqueles que ja foram detectados 3 semanas antes
        set2 = dta[(dta[col_warn_s1] == 1) & (dta[col_warn_s2].shift(-3) == 0) & (dta[col_warn_s2].shift(-2) == 1)].index + 2
        # encontra todos os surtos que estão sendo detectados 1 semanas antes e excluindo os casos anteriores
        set1 = dta[(dta[col_warn_s1] == 1) & 
                   (dta[col_warn_s2].shift(-3) == 0) & 
                   (dta[col_warn_s2].shift(-2) == 0) & 
                   (dta[col_warn_s2].shift(-1) == 1)].index + 1
        # encontra todos os surtos que estão sendo detectados namesma semanas e excluindo os casos anteriores
        set0 = dta[(dta[col_warn_s1] == 1) & 
                   (dta[col_warn_s2] == 1) & 
                   (dta[col_warn_s2].shift(-3) == 0) & 
                   (dta[col_warn_s2].shift(-2) == 0) & 
                   (dta[col_warn_s2].shift(-1) == 0)].index
        
        # non surge weeks without alarm (Identify True Negatives (TN))
        set_tn1 = dta[((dta[col_warn_s2_with_consec] == 0) &
                       (dta[col_warn_s2_with_consec].shift(-1).fillna(0) == 0) &
                      (dta[col_warn_s2_with_consec].shift(-2).fillna(0) == 0) &
                      (dta[col_warn_s2_with_consec].shift(-3).fillna(0) == 0)) & # aqui deve usar a coluna com os avisos em blocos corrigidos col_warn_s2
                       (dta[col_warn_s1] == 0)].index
        
        # non-surge with alarms in the same or previous weeks
        set_fp1 = dta[((dta[col_warn_s2_with_consec] == 0) &
                       (dta[col_warn_s2_with_consec].shift(-1).fillna(0) == 0) &
                      (dta[col_warn_s2_with_consec].shift(-2).fillna(0) == 0) &
                      (dta[col_warn_s2_with_consec].shift(-3).fillna(0) == 0)) & 
                       (dta[col_warn_s1] == 1)].index

        # Warnings in PHC right after an AIH warning (possibly not anticipated but concurrent)
        set1_after = dta[(dta[col_warn_s1].shift(-1) == 1) & (dta[col_warn_s2] == 1)].index
        
       

        # Compute counts of warnings at different lead times
        n3 = len(set3)
        n2 = len(set(set2) - set(set3))
        n1 = len((set(set1) - set(set3)) - set(set2))
        n0 = len(((set(set0) - set(set3)) - set(set2)) - set(set1))
        n1_after = len((((set(set1_after) - set(set3)) - set(set2)) - set(set1)) - set(set0))
        n_tn1 = len(set_tn1) 
        n_fp1 = len(set_fp1)

        # Drop all anticipated and concurrent warnings to count missed ones
        ind_drop = set(set3) | set(set2) | set(set1) | set(set0) #| set(set1_after)
        missed = dta.drop(index=ind_drop)[col_warn_s2].sum()
        
        TP = n3 + n2 + n1 + n0

        TP_ = TP*3 + sum(dta[col_warn_s2_with_consec])
        
        # outra forma de calcular o FN
        #set_FN = dta[(dta[col_warn_s2] == 1) &  
        #           (dta[col_warn_s1] == 0) & 
        #           (dta[col_warn_s1].shift(3).fillna(0) == 0) & 
        #           (dta[col_warn_s1].shift(2).fillna(0) == 0) & 
        #           (dta[col_warn_s1].shift(1).fillna(0) == 0)].index
        
        
        # Create the results dictionary for this region
        data = {
            col_code: [code],
            'n3': [n3],
            'n2': [n2],
            'n1': [n1],
            'n0': [n0],
            'n1_after': [n1_after],
            'missed': [missed],
            'TP': [TP],
            'TP_':[TP_],
            'TN1': [n_tn1], # True Negative
            'FP1': [n_fp1], # False Positive
            'FN': missed,
            'total_aih_warning': [dta[col_warn_s2].sum()]
        }

        # Append to results list
        data_output = pd.DataFrame(data)
        lst_count.append(data_output)

    # Combine all results into a single DataFrame
    df_warning_count = pd.concat(lst_count, ignore_index=True)
    
    return df_warning_count


def summarize_performance(df_warning_count):
    """
    Summarizes performance metrics including Sensitivity, Specificity, PPV, F1-score, POD, and FPR.
    
    Parameters:
    df_warning_count (pd.DataFrame): Dataframe containing warning counts and classification metrics.
    
    Returns:
    pd.DataFrame: Summary table with performance metrics.
    """
    # Total AIH Warnings
    total_warnings = df_warning_count.total_aih_warning.sum()
    
    # Early Detection (1 to 3 weeks)
    early_count = df_warning_count.n3.sum() + df_warning_count.n2.sum() + df_warning_count.n1.sum()
    early_rate = round((early_count * 100) / total_warnings, 1)
    
    # Timely Detection (0 week)
    timely_count = df_warning_count.n0.sum()
    timely_rate = round((timely_count * 100) / total_warnings, 1)
    
    # Missed Warnings
    missed_count = df_warning_count.missed.sum()
    missed_rate = round((missed_count * 100) / total_warnings, 1)

    # Timeliness 
    s3 = df_warning_count.n3.sum()
    s3_rate = round((s3 * 100) / total_warnings, 1)
    s2 = df_warning_count.n2.sum()
    s2_rate = round((s2 * 100) / total_warnings, 1)
    s1 = df_warning_count.n1.sum()
    s1_rate = round((s1 * 100) / total_warnings, 1)
    s0 = df_warning_count.n0.sum()
    s0_rate = round((s0 * 100) / total_warnings, 1)

    u3 = np.exp(0.5 * 3)
    u2 = np.exp(0.5 * 2)
    u1 = np.exp(0.5 * 1)
    u0 = np.exp(0.5 * 0)
    
    timeliness = (u3*s3 + u2*s2 + u1*s1 + u0*s0) / (u3 + u2 + u1 + u0)

    max_time = u3*total_warnings / (u3 + u2 + u1 + u0)

    # Detection rate

    R = (early_count + timely_count)/ total_warnings

    # Score of balance between timeliness and missing
    S = timeliness*R
    
    # Classification Metrics
    TP = df_warning_count.TP.sum()
    FN = df_warning_count.FN.sum()
    TN = df_warning_count.TN1.sum()
    FP = df_warning_count.FP1.sum()
    TP_ = df_warning_count['TP_'].sum()

    
    # Sensitivity (Recall)
    Sensitivity = round((TP / (TP + FN)) * 100, 1) if (TP + FN) > 0 else 0
    
    # Specificity
    Specificity = round((TN / (TN + FP)) * 100, 1) if (TN + FP) > 0 else 0
    
    # Positive Predictive Value (PPV)
    PPV = round((TP_ / (TP_ + FP)) * 100, 1) if (TP + FP) > 0 else 0

    # Positive Predictive Value (PPV)
    NPV = round((TN / (TN + FN)) * 100, 1) if (TN + FN) > 0 else 0
    
    # F1-score
    F1_score = round(2 * (TP / (TP + FN)) * (TP / (TP + FP)) / ((TP / (TP + FN)) + (TP / (TP + FP))), 2) if ((TP / (TP + FN)) + (TP / (TP + FP))) > 0 else 0
    
    Precision = round(((TP + TN) / (TP + TN + FP + FN) * 100), 1) if (TP + TN + FP + FN) > 0 else 0

    # Probability of Detection (POD)
    POD = round(((early_count + timely_count + df_warning_count.n1_after.sum()) / total_warnings) * 100, 1)
    
    # False Positive Rate (FPR)
    FPR = round((FP / (FP + TN)) * 100, 1) if (FP + TN) > 0 else 0

    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        "Metric": ["Total Warnings", "Early Detection (1-3 weeks)", "Timely Detection (0 weeks)", "Missed Warnings",
                   "Sensitivity ", "Specificity", "PPV",  "NPV", "POD", "FPR","Precision (%)",'Timeliness', 'Score',
                  '3 weeks earlier','2 weeks earlier','1 week earlier', 'Same week (0)', 'Max_timeliness','TP + FN', 'TN + FP',
                  'TP_ + FP', 'TN + FN', 'TP', 'FN', 'TN', 'FP', 'TP_','n3','n2','n1','n0'],
        "Value": [total_warnings, f"{early_count} ({early_rate}%)", f"{timely_count} ({timely_rate}%)", f"{missed_count} ({missed_rate}%)",
                   f"{Sensitivity}%", f"{Specificity}%", f"{PPV}%", f"{NPV}%",f"{POD}%", f"{FPR}%", f"{Precision}%",f"{timeliness}",f"{S}",
                 f"{s3_rate}", f"{s2_rate}", f"{s1_rate}", f"{s0_rate}",  f"{max_time}",f"{TP + FN}", f"{TN + FP}",f"{TP_ + FP}",f"{TN + FN}",
                 f"{TP}",f"{FN}",f"{TN}",f"{FP}",f"{TP_}",f"{s3}",f"{s2}", f"{s1}",f"{s0}"]
    })
    
    return summary_df


# Youndex
def opt_evi_par(m, c, set_muni):

    dtf = func(m, set_muni.atend_ivas.to_numpy(), 2, 8, c)

    set_muni = set_muni.reset_index(drop=True).copy()
    set_muni["evi_t1_t"] = dtf["evi_t1_t"]
    set_muni["ind"] = dtf["ind"]

    max_value = np.nanmax(set_muni.loc[np.isfinite(set_muni['evi_t1_t']), 'evi_t1_t'])
    set_muni['evi_t1_t'].replace([np.inf, -np.inf], max_value, inplace=True)

    set_muni["sinal_evi_ivas"] = set_muni["ind"].astype(int)

    dta1 = set_muni[
        ['co_uf', 'nm_uf', 'nm_municipio', 'co_ibge', 'year_week',
         'atend_totais', 'atend_ivas',
         'warning_final_mem_surge_01', 'mem_surge_01_correct_with_consec',
         'sinal_evi_ivas']
                   ]

    df_warning_count = antici_count(
        dta1,
        'sinal_evi_ivas',
        'warning_final_mem_surge_01',
        'mem_surge_01_correct_with_consec',
        'co_ibge'
    )
    

    performance_summary = summarize_performance(df_warning_count)

    # --- extract metrics safely ---
    perf = performance_summary.set_index("Metric")["Value"]

    sens = float(perf["Sensitivity "].strip('%')) / 100
    spec = float(perf["Specificity"].strip('%')) / 100

    n3 = int(perf['n3'])
    n2 = int(perf['n2'])
    n1 = int(perf['n1'])
    n0 = int(perf['n0'])

    J = sens + spec - 1

    w = 0.75*n3 + 0.20*n2 + 0.04*n1 + 0.01*n0

    return {
        "m": m,
        "c": c,
        "sensitivity": sens,
        "specificity": spec,
        "youden_J": J,
        "n3": n3,
        "n2": n2,
        "n1": n1,
        "n0": n0,
        "w": w
    }




def grid_search_m_c(m_values, c_values, set_muni):

    results = []

    for m in m_values:
        for c in c_values:
            try:
                res = opt_evi_par(m, c, set_muni)
                results.append(res)
            except Exception as e:
                print(f"Failed for m={m}, c={c}: {e}")

    return pd.DataFrame(results)

