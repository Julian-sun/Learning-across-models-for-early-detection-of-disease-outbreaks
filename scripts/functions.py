# Author
# Juliane Oliveira


# packs

import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from datetime import datetime


########### Start the game

#### Function to process data

def year_week(y, w):
    """
    Convert year and week number to a datetime object representing the first day of that week.

    Parameters:
    y (int or str): The year.
    w (int or str): The ISO week number.

    Returns:
    datetime: A datetime object for the first day of the specified week.
    """
    return datetime.strptime(f'{y} {w} 1', '%G %V %w')


def year_week_ts(df, epiweek_col='epiweek', year_col='ano'):
    """
    Add columns to the DataFrame for ISO year-week, a datetime object for the first day of that week,
    and a string representation of the year-week.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing year and week columns.
    epiweek_col (str): The name of the column in df representing the week number. Default is 'epiweek'.
    year_col (str): The name of the column in df representing the year. Default is 'ano'.

    Returns:
    pd.DataFrame: The DataFrame with additional columns:
        - 'year_week': ISO year-week in 'YYYY-WW' format.
        - 'year_week_ts': datetime object for the first day of the specified week.
        - 'year_week_str': String representation of the year-week in 'YYYY-WW' format.
    """
    # Create the 'year_week' column in the format 'YYYY-WW'
    df['year_week'] = df.apply(
        lambda row: f"{row[year_col]}-{int(row[epiweek_col]):02d}", axis=1
    )

    # Create the 'year_week_ts' column using the year_week function
    df['year_week_ts'] = df.apply(
        lambda row: year_week(row[year_col], row[epiweek_col]), axis=1
    )

    # Create the 'year_week_str' column as a string in 'YYYY-WW' format
    df['year_week_str'] = df['year_week_ts'].apply(
        lambda x: x.strftime('%G-%V')
    )

    return df


# Create list of cities dataframe
def lst_dfs_cities(df, city_code_col='co_ibge', epiweek_date_col='year_week_str', ili_col='atend_ivas', otc_col='unidades_gripal'):
    """
    Create a list of DataFrames for each unique city in the input DataFrame,
    with additional rolling mean and lag features.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the necessary columns.
    city_code_col (str): Column name for city codes. Default is 'co_ibge'.
    epiweek_date_col (str): Column name for epidemiological week dates. Default is 'year_week_str'.
    ili_col (str): Column name for ILI (Influenza-Like Illness) counts. Default is 'atend_ivas'.
    otc_col (str): Column name for OTC (Over-The-Counter) medication counts. Default is 'unidades_gripal'.

    Returns:
    list: A list of DataFrames, each corresponding to a unique city, with additional computed columns:
        - Rolling mean and log-transformed columns for 'ili_col' and 'otc_col' over a 4-week window.
        - Lagged values and their logarithms for 'ili_col' and 'otc_col' for lags 1 to 4 weeks.
    """
    lst_dfs_cities = []
    
    for muni in df[city_code_col].unique():
        # Select data for a specific municipality
        set_muni = df[df[city_code_col] == muni]

        # Sort by the epidemiological week date column
        set_muni = set_muni.sort_values(by=epiweek_date_col)

        # Calculate 4-week rolling means and rename columns
        ili_4_col = f'{ili_col}_4'
        otc_4_col = f'{otc_col}_4'
        set_muni = set_muni.assign(ili_4=set_muni[ili_col].rolling(window=4).mean())
        set_muni = set_muni.rename(columns={'ili_4': ili_4_col})
        set_muni = set_muni.assign(otc_4=set_muni[otc_col].rolling(window=4).mean())
        set_muni = set_muni.rename(columns={'otc_4': otc_4_col})

        # Calculate logarithms of rolling means and rename columns
        set_muni = set_muni.assign(ili_4_log=np.log(set_muni[ili_4_col]))
        set_muni = set_muni.rename(columns={'ili_4_log': f'{ili_col}_4_log'})
        set_muni = set_muni.assign(otc_4_log=np.log(set_muni[otc_4_col]))
        set_muni = set_muni.rename(columns={'otc_4_log': f'{otc_col}_4_log'})

        # Calculate lagged values and their logarithms
        for lag in range(1, 5):
            set_muni = set_muni.assign(**{f'{otc_col}_4_lag_{lag}': set_muni[otc_4_col].shift(lag)})
            #set_muni = set_muni.assign(**{f'{otc_col}_4_log_lag_{lag}': np.log(set_muni[f'{otc_col}_4_lag_{lag}'])})
            
            set_muni = set_muni.assign(**{f'{ili_col}_4_lag_{lag}': set_muni[ili_4_col].shift(lag)})
            #set_muni = set_muni.assign(**{f'{ili_col}_4_log_lag_{lag}': np.log(set_muni[f'{ili_col}_4_lag_{lag}'])})

        # Fill NaN values with 0
        set_muni = set_muni.fillna(0)

        # Append the DataFrame to the list
        lst_dfs_cities.append(set_muni)
        
    return lst_dfs_cities

#Harmonic terms

# Function to perform FFT and reconstruct the time series
def best_fft_reconstruction(data_values, period=52):
    # Convert to numpy array for FFT
    S = np.array(data_values)

    # Perform FFT
    N = period
    T = 1.0  # Assuming the time step is 1 (weekly data)
    yf = fft(S)

    best_S_reconstructed = None
    best_tsig = None
    yf_filtered_best = None

    for tsig in [0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]:
        # Threshold for significant components (can be adjusted)
        threshold = np.max(np.abs(yf)) * tsig

        # Zero out insignificant components
        yf_filtered = yf.copy()
        yf_filtered[np.abs(yf) < threshold] = 0

        # Perform inverse FFT to reconstruct the time series
        S_reconstructed = ifft(yf_filtered).real

        # Update the best reconstruction if variance is greater than 0
        if S_reconstructed.var() > 0.001:
            best_S_reconstructed = S_reconstructed
            best_tsig = tsig
            yf_filtered_best = yf_filtered
            break

    return best_S_reconstructed, best_tsig, yf_filtered_best

# Define the Fourier series function based on the extracted coefficients
def fourier_series(t, a0, an, bn, N):
    result = a0
    for n, (a, b) in enumerate(zip(an, bn), start=1):
        result += a * np.cos(2 * np.pi * n * t / N) + b * np.sin(2 * np.pi * n * t / N)
    return result

def harmonic(list_dfs, var):
    """
    Add harmonic features to each DataFrame in the input list.
    
    Parameters:
    - list_dfs (list): List of DataFrames to be processed.
    - variable to be used, ex: 'atend_ivas_4' 
    
    Returns:
    list: The input list with each DataFrame containing additional harmonic features.
    """
    lst_dfs = []
    
    for df in list_dfs:
        # Time series data
        data_values = df[var][0:52].to_numpy()

        # Get the best reconstructed time series and the corresponding tsig
        best_S_reconstructed, best_tsig, yf_filtered_best = best_fft_reconstruction(data_values, period=52)
        
        # Check if yf_filtered_best is None and skip processing if so
        if yf_filtered_best is None:
            print("Skipping DataFrame due to yf_filtered_best being None",  df['co_ibge'].iloc[0])
            continue

        # Extract the significant coefficients from the best filtered yf
        N = 52
        a0 = np.real(yf_filtered_best[0]) / N
        an = 2.0 / N * np.real(yf_filtered_best[1:N//2])
        bn = -2.0 / N * np.imag(yf_filtered_best[1:N//2])

        # Set the desired number of decimal places
        decimal_places = 6
        

        # Construct the Fourier series equation with more precision
        equation = f"S(t) = {a0:.{decimal_places}f}"
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= 1e-6 or np.abs(b) >= 1e-6:  # Exclude small coefficients
                equation += f" + ({a:.{decimal_places}f}) * cos(2π * {n} * t / {N}) + ({b:.{decimal_places}f}) * sin(2π * {n} * t / {N})"
        
        # Compute the values of the Fourier series for the same time points
        time_points = np.arange(len(data_values))
        fourier_series_values = fourier_series(time_points, a0, an, bn, N)


        data_values = df[var].to_numpy()
        # Compute the values of the Fourier series for the same time points
        time_points = np.arange(len(data_values))
        fourier_series_values = fourier_series(time_points, a0, an, bn, N)

        #set_muni = set_muni.assign(harmonics = fourier_series_values - a0)
        df = df.assign(S_t_fou = fourier_series_values)

        t = np.arange(len(df))

        # Define a threshold for significant terms
        threshold = 1e-6


        # Add columns for each significant Fourier term
        df['a0'] = a0
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= threshold:
                df[f'cos_{n}'] = a * np.cos(2 * np.pi * n * t / N)
            if np.abs(b) >= threshold:
                df[f'sin_{n}'] = b * np.sin(2 * np.pi * n * t / N)

        # Sum up the significant Fourier terms to get the reconstructed series
        df['Reconstructed'] = df['a0'] + df.filter(like='cos').sum(axis=1) + df.filter(like='sin').sum(axis=1)
        
        df = df.assign(Reconstructed_log = np.log(df.Reconstructed))
        
        df.Reconstructed_log = df.Reconstructed_log.fillna(0)
        
        v1 = df.loc[~df['Reconstructed_log'].isin([np.inf,-np.inf]),'Reconstructed_log'].max()
        v2 = df.loc[~df['Reconstructed_log'].isin([np.inf,-np.inf]),'Reconstructed_log'].min()
        df['Reconstructed_log'] = df['Reconstructed_log'].replace([np.inf,-np.inf], [v1,v2])
        
        lst_dfs.append(df)

    return lst_dfs



### Remove isolated warnings and group consecutive ones

def clean_warning_column(
    data: pd.DataFrame,
    group_col="co_ibge",
    time_col="year_week",
    warning_col="mem_surge_01"
):

    lst = []

    for code in data[group_col].unique():

        set_region = data[data[group_col] ==  code]
        
        set_region = set_region.sort_values(by = time_col)

        # Create shifted columns to check neighbors
        set_region['prev1'] = set_region[warning_col].shift(1, fill_value=0)
        set_region['next1'] = set_region[warning_col].shift(-1, fill_value=0)
        set_region['prev2'] = set_region[warning_col].shift(2, fill_value=0)
        set_region['next2'] = set_region[warning_col].shift(-2, fill_value=0)

        # Identify isolated warnings (a `1` surrounded by `0`s)
        set_region['is_isolated'] = ( (set_region[warning_col] == 1) &  # Current value is 1
                            (set_region['prev1'] == 0) &        # Previous value is 0
                            (set_region['next1'] == 0) &        # Next value is 0
                            (set_region['prev2'] == 0) &        # Previous 2 value is 0 
                            (set_region['next2'] == 0)          # Next 2 value is 0
                          )

        # Replace isolated warnings with 0
        set_region['cleaned_warning'] = set_region[warning_col].where(~set_region['is_isolated'], 0)

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


def insert_outbreak(series, start_week, duration, k, lognorm_mean=0, lognorm_sd=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    series = series.copy()
    n = len(series)

    sigma = np.std(series)

    # total outbreak size
    outbreak_size = rng.poisson(k * sigma)

    if outbreak_size == 0:
        return series, {"start": start_week, "end": start_week, "size": 0}

    # lognormal temporal distribution
    times = rng.lognormal(lognorm_mean, lognorm_sd, outbreak_size)

    # scale to chosen duration
    offsets = np.floor((times / times.max()) * (duration-1)).astype(int)

    outbreak_weeks = start_week + offsets
    outbreak_weeks = outbreak_weeks[outbreak_weeks < n]

    for w in outbreak_weeks:
        series[w] += 1

    return series, {
        "start": start_week,
        "end": int(start_week + duration),
        "size": int(outbreak_size),
        'k':k,
        "duration": duration
    }


def generate_replicate(series, max_outbreaks=2, min_gap=4, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    n = len(series)
    new_series = series.copy()

    # number of outbreaks
    K = rng.integers(1, max_outbreaks + 1)

    outbreaks = []

    attempts = 0
    max_attempts = 200

    while len(outbreaks) < K and attempts < max_attempts:

        start = rng.integers(415, n-10) # aqui muda as o inicio da semana
        duration = rng.integers(4, 11)
        end = start + duration

        valid = True

        # check overlap + minimum gap
        for ob in outbreaks:

            existing_start = ob["start"]
            existing_end = ob["end"]

            if not (end + min_gap < existing_start or start > existing_end + min_gap):
                valid = False
                break

        if valid:

            k = rng.uniform(2, 10)

            new_series, info = insert_outbreak(
                new_series,
                start_week=start,
                duration=duration,
                k=k,
                rng=rng
            )

            outbreaks.append(info)

        attempts += 1

    return new_series, outbreaks
