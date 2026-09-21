import polars as pl


def ensemble(df: pl.DataFrame, pattern="^sinal_.*_ivas$") -> pl.DataFrame:
    cols = df.select(pl.col(pattern)).columns
    suffix_atend = cols[0].split("_")[2]
    cols = [c for c in cols if c != f"sinal_otc_{suffix_atend}"]

    df = df.with_columns(
        pl.sum_horizontal(
            *[pl.col(col) for col in cols], ignore_nulls=True
        ).alias("sum_signs")
    ).with_columns(
        pl.sum_horizontal(
            *[pl.col(col).is_not_null().cast(pl.Int64) for col in cols]
        ).alias("len_valid")
    ).with_columns(
        (pl.col("sum_signs") > (pl.col("len_valid") / 2))
        .cast(pl.Int8).alias(f"sinal_ens_{suffix_atend}")
    ).drop("sum_signs", "len_valid")
    
    return df
