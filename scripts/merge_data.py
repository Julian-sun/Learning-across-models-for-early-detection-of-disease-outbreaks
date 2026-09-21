from functools import reduce
import os
from pathlib import Path
import polars as pl
import numpy as np
from funcs import ensemble


aps_raw = os.getenv(
    "APS_ATUAL",
    "/opt/storage/shared/aesop/aesop_shared/atual/base_aps_atual.parquet"
)

aesop_mun = os.getenv(
    "AESOP_VIS",
    "/opt/storage/refined/aesop/visualization/aesop_2025_08_05_mun.parquet"
)


excess_parquet = os.getenv(
    "AESOP_VIS_EXCESS",
    "/opt/storage/refined/aesop/visualization/excess.parquet"
)


data = Path(__file__).absolute().parent.parent / "data"
(data / 'final').mkdir(exist_ok=True, parents=True)


"""
aps = pl.read_parquet(
    aps_raw,
    columns=["co_ibge", "ano", "epiweek", "atend_ivas", "atend_totais"]
).with_columns(
    pl.concat_str(
        pl.col("ano"),
        pl.col("epiweek").map_elements(lambda x: f"{int(x):02d}",
                                       return_dtype=pl.String),
        separator="-"
    ).alias("year_week"),
).drop("ano", "epiweek")
"""

aesop_df = pl.read_parquet(
    aesop_mun,
    columns=[
        "dqi",
        "co_ibge",
        "year_week",
        "atend_ivas",
        "atend_totais",
        "P_growth_aps_ivas",
        "P_growth_otc_ivas",
        "sinal_otc_ivas",
        "exc_atend_ivas",
     ]
)

excess_df = pl.read_parquet(
    excess_parquet,
    columns=[
        "co_ibge",
        "epiyear",
        "epiweek",
        "expected_atend_ivas",
        "exc_atend_ivas",
        "ci_lower_95_atend_ivas",
        "ci_upper_95_atend_ivas",
    ]
).with_columns(
    pl.concat_str(
        pl.col("epiyear"),
        pl.col("epiweek").map_elements(lambda x: f"{int(x):02d}",
                                       return_dtype=pl.String),
        separator="-"
    ).alias("year_week")
)

ears = pl.read_parquet(data / "raw/202322_202522_ears_allcount.parquet")

mmaing = pl.read_csv(
    data / "raw/EWS_municipios_completo_sem_DQI.csv",
    separator=";"
)

evi = pl.read_parquet(
    data / "raw/202322_202522_evi_ivas.parquet"
).with_columns(
    pl.col("alarm").cast(pl.Int8)
).rename({"alarm": "sinal_evi_ivas", "upperbound": "limite_evi_ivas"}).select(
    "co_ibge", "year_week", pl.col(r"^sinal_.*_ivas$"), pl.col(r"^limite_.*_ivas$")
)


ears = ears.with_columns(
    pl.concat_str(
        pl.col("ano"),
        pl.col("epiweek").map_elements(lambda x: f"{int(x):02d}",
                                       return_dtype=pl.String),
        separator="-"
    ).alias("year_week"),
    pl.col("alarm_ears").map_elements(
        lambda x: 1 if x == "Sim" else 0 if x == "Não" else None,
        return_dtype=pl.Int8
    )
).rename({"alarm_ears": "sinal_ears_ivas", "upperbound": "limite_ears_ivas"}).select(
    "co_ibge", "year_week", pl.col(r"^sinal_.*_ivas$"), pl.col(r"^limite_.*_ivas$")
)


mmaing = mmaing.with_columns(
    pl.concat_str(
        pl.col("ano"),
        pl.col("epiweek").map_elements(lambda x: f"{int(x):02d}",
                                       return_dtype=pl.String),
        separator="-"
    ).alias("year_week"),
    pl.col("EWS_MMAING").map_elements(
        lambda x:
        1 if x == "Sim" else
        0 if x == "Não" else
        None if x == np.nan else
        int(x),
        return_dtype=pl.Int8
    ).alias("sinal_mmaing_ivas")
).filter(
    (pl.col("year_week") >= "2023-22") &
    (pl.col("year_week") <= "2025-22")
).rename({"upperbound": "limite_mmaing_ivas"}).select(
    "co_ibge", "year_week", pl.col(r"^sinal_.*_ivas$"), pl.col(r"^limite_.*_ivas$")
)

df_merge = reduce(
    lambda df0, df1:
        df0.join(
            df1,
            on=list(set(df0.columns) & set(df1.columns)),
            how="left"
        ), [ears, evi, mmaing, aesop_df]
)
df_merge = df_merge.join(excess_df, on=["co_ibge", "year_week"], how="left")

"""
df_merge = merge_all_dfs([ears, evi, mmaing, aps, excess_df, aesop_df])
"""

df_merge = df_merge.select(
    "dqi",
    "co_ibge",
    "year_week",
    pl.col(r"^sinal_.*_ivas$"),
    "P_growth_aps_ivas",
    pl.col(r"^limite_.*_ivas$"),
    "atend_ivas",
    "atend_totais",
    "expected_atend_ivas",
    "exc_atend_ivas",
    "ci_lower_95_atend_ivas",
    "ci_upper_95_atend_ivas",
)

df_merge = ensemble(df_merge, pattern="^sinal_.*_ivas$")

df_merge = df_merge.with_columns(
    pl.when(
        ((pl.col("dqi") == "Apto") & (pl.col("sinal_ens_ivas") == 1)) |
        (pl.col("sinal_otc_ivas") == 1)
    ).then(1).otherwise(0).alias("sinal_aps_ivas")
)

df_merge.write_parquet(data / "final/202322_202522_ears_evi_mmaing.parquet")
