import polars as pl


def inspect_lazy(lf: pl.LazyFrame) -> None:
    """
    Print the shape of a Polars LazyFrame in a memory-efficient manner.

    This function computes the number of rows using a lazy row-count
    aggregation (`pl.len()`) and retrieves the number of columns from
    the resolved schema without materializing the full dataset.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to inspect.

    Notes
    -----
    - The row count triggers execution of the lazy query plan,
      but avoids collecting all columns into memory.
    - The column count is obtained from the schema metadata and
      does not require data materialization.
    - Intended for debugging and validation of large lazy pipelines.

    """
    n_rows = lf.select(pl.len()).collect().item()
    n_cols = len(lf.collect_schema())
    print(f"Rows: {n_rows:,}")
    print(f"Columns: {n_cols}")
