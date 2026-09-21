import os
from pathlib import Path
import polars as pl


data = Path(__file__).absolute().parent.parent / "data"
(data / 'transient').mkdir(exist_ok=True, parents=True)


raw_aps = os.getenv(
    'APS_ATUAL',
    '/opt/storage/shared/aesop/aesop_shared/atual/base_aps_atual.parquet'
)

(
    pl.read_parquet(
        raw_aps,
        columns=[
            'co_ibge',
            'ano',
            'epiweek',
            'atend_ivas',
            'atend_totais'
        ]
    )
    .filter(pl.col('ano').is_between(2020, 2025))
    .with_columns(
        pl.concat_str(
            pl.col('ano'),
            pl.col('epiweek')
            .map_elements(lambda x: f'{int(x):02d}', return_dtype=pl.Utf8),
            separator='-').alias('year_week')
    )
    .sort('co_ibge', 'year_week')
    .write_parquet(data / 'transient/aesop_atend_ivas_total_2020_2025.parquet')
)
