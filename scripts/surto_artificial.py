from pathlib import Path
from epiweeks import Week
import numpy as np
import pandas as pd

from functions import generate_replicate


path = Path("/opt/storage/shared/aesop/aesop_shared/ensamble_modelling/")
outbreak_path = path / "series_sinteticas_new/outbreaks_metadata/"
sinteticas_path = path / "series_sinteticas_new/"
outbreak_path.mkdir(parents=True, exist_ok=True)

last_aesop_with_mem = max(
    path.glob("aesop_*_with_MEM.parquet"),
    key=lambda x: x.stat().st_mtime
)

columns = [
    "co_ibge",
	"year_week",
	"atend_ivas",
	"warning_final_mem_surge_01",
]

df = pd.read_parquet(
    last_aesop_with_mem,
    columns=columns
)

df["epidemi_cal_start"] = df.apply(
    lambda r:
    Week(*map(int, r.year_week.split("-"))).startdate(),
    axis=1
)


df_cut = df[df.year_week <= "2025-32"]

rng = np.random.default_rng(42)
replicas = 42


for co_ibge in df_cut["co_ibge"].unique():
    set_muni = df_cut[df_cut.co_ibge == co_ibge].copy()
    baseline = set_muni.atend_ivas.values

    replicates, outbreak_metadata = [], []
    for i in range(replicas):
        new_series, outbreak = generate_replicate(baseline, rng=rng)
        for outbreak_d in outbreak:
            year_week_start, year_week_end, epidemi_cal_start = (
                set_muni.iloc[outbreak_d["start"]]["year_week"],
                set_muni.iloc[outbreak_d["end"]]["year_week"],
                set_muni.iloc[outbreak_d["start"]]["epidemi_cal_start"],
            )
            outbreak_d["co_ibge"] = co_ibge
            outbreak_d["start"], outbreak_d["end"], outbreak_d["epidemi_cal_start"] = (
                year_week_start, year_week_end, epidemi_cal_start
            )
        replicates.append(new_series)
        outbreak_metadata.append(pd.concat([pd.DataFrame(data, index=[1]) for data in outbreak]))

    replicates_df = pd.DataFrame(replicates).T
    replicates_df.columns = [f"replicate_{i}" for i in range(len(replicates))]
    replicates_df["co_ibge"] = co_ibge
    set_muni = set_muni.assign(replicate_3 = replicates_df.replicate_3)

    replicates_df = pd.concat(
        [replicates_df, set_muni[["year_week", "epidemi_cal_start"]]], axis=1
    )
    replicates_df.to_parquet(sinteticas_path / f"{co_ibge}.parquet")

    pd.concat(outbreak_metadata) \
        .reset_index(drop=True) \
        .to_parquet(outbreak_path / f"{co_ibge}.parquet")
