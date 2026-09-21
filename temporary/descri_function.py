import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from matplotlib import pyplot as plt
from matplotlib_venn import venn3

import statsmodels.api as sm

import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m), round(m-h), round(m+h)

def mean_confidence_interval2(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,4), round(m-h,4), round(m+h,4)


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


# FUNCTION TO ASSIGN INTENSITY

def classify_block(values):
    
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
    
    elif (values <= base).any():
        return 'baseline'
    
    return np.nan


#########
### Remove isolated warnings and group consecutive ones

def clean_warning_column(data, col_code_region, col_of_dates,col_of_warnings):

    lst = []

    for code in data[col_code_region].unique():

        set_region = data[data[col_code_region] ==  code]
        
        set_region = set_region.sort_values(by = 'year_week')

        # Create shifted columns to check neighbors
        set_region['prev1'] = set_region[col_of_warnings].shift(1, fill_value=0)
        set_region['next1'] = set_region[col_of_warnings].shift(-1, fill_value=0)
        set_region['prev2'] = set_region[col_of_warnings].shift(2, fill_value=0)
        set_region['next2'] = set_region[col_of_warnings].shift(-2, fill_value=0)

        # Identify isolated warnings (a `1` surrounded by `0`s)
        set_region['is_isolated'] = ( (set_region[col_of_warnings] == 1) &  # Current value is 1
                            (set_region['prev1'] == 0) &        # Previous value is 0
                            (set_region['next1'] == 0) &        # Next value is 0
                            (set_region['prev2'] == 0) &        # Previous 2 value is 0 
                            (set_region['next2'] == 0)          # Next 2 value is 0
                          )

        # Replace isolated warnings with 0
        set_region['cleaned_warning'] = set_region[col_of_warnings].where(~set_region['is_isolated'], 0)

        # Create a helper column to find groups of consecutive 1s or 1s separated by one 0
        set_region = set_region.assign(group = ((set_region['cleaned_warning'] == 1) & 
                                    (~set_region['cleaned_warning'].shift().fillna(0).astype(bool)) | 
                                    ((set_region['cleaned_warning'] == 1) & 
                                     set_region['cleaned_warning'].shift(2).fillna(0).astype(bool))).cumsum())

        # Identify groups of events (1s including those separated by 1 week)
        set_region['event'] = ((set_region['cleaned_warning'] == 1) | 
                     ((set_region['cleaned_warning'] == 0) & 
                      (set_region['cleaned_warning'].shift(-1) == 1) & 
                      (set_region['cleaned_warning'].shift() == 1))).astype(int)

        # Collapse each event into a single identifier
        set_region['final_event'] = (set_region['event'] != set_region['event'].shift()).cumsum() * set_region['event']

        set_region = set_region.assign(warning_final = set_region.final_event - set_region.final_event.shift().fillna(0))

        # Replace values: > 0 becomes 1, <= 0 becomes 0
        set_region['warning_final'] = (set_region['warning_final'] > 0).astype(int)
        
        lst.append(set_region)
    
    data = pd.concat(lst)
    
    return data
