"""
ssyk2012_daioe_yr_regions.py.
----------------------------
Merges DAIOE AI-exposure scores with SCB employment data across
SSYK2012 occupational levels (1-4), counties, sex, and years.

Pipeline:
    1.  Load DAIOE and SCB lazily
    2.  Compute 1/3/5-year employment changes (SCB)
    3.  Derive SSYK2012 hierarchy codes (DAIOE)
    4.  Align DAIOE years to latest SCB coverage
    5.  Join DAIOE with SCB SSYK4 counts
    6.  Aggregate DAIOE metrics to all SSYK levels (1-4)
    7.  Build 1-5 exposure level columns
    8.  Final merge and parquet export
"""  # noqa: D205

from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Paths & Sources
# ---------------------------------------------------------------------------

ROOT = Path.cwd().resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DAIOE_SOURCE = (
    "https://raw.githubusercontent.com/joseph-data/07_translate_ssyk/main/"
    "03_translated_files/daioe_ssyk2012_translated.csv"
)
SCB_SOURCE = (
    "https://raw.githubusercontent.com/joseph-data/AI_Econ_daioe_years_regions/daioe_pull/"
    "data/processed/ssyk12_aggregated_ssyk4_to_ssyk1.parquet"
)

OUTPUT_PATH = DATA_DIR / "daioe_scb_years_all_levels.parquet"

# First year of SSYK2012 publication
SSYK12_START_YEAR = 2014


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AggConfig:
    """Configuration for DAIOE metric aggregation."""

    weight_col: str = "total_count"
    prefix: str = "daioe_"
    add_percentiles: bool = True
    pct_scale: int = 100
    descending: bool = False
    # Exposure level percentile boundaries (quintile-style)
    pct_l1: int = 20
    pct_l2: int = 40
    pct_l3: int = 60
    pct_l4: int = 80


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------


def aggregate_daioe_level(
    lf: pl.LazyFrame,
    code_col: str,
    level_label: str,
    cfg: AggConfig | None = None,
) -> pl.LazyFrame:
    """
    Aggregate DAIOE metrics by SSYK level with optional weighted averages
    and within-year percentile ranks.

    Args:
        lf:           Input LazyFrame (must contain DAIOE columns and weight_col).
        code_col:     Column name for SSYK code (e.g. "code_1", "code_2").
        level_label:  Label for the aggregation level (e.g. "SSYK1", "SSYK2").
        cfg:          Aggregation configuration. Defaults to AggConfig().

    """  # noqa: D205
    if cfg is None:
        cfg = AggConfig()

    daioe_cols = [c for c in lf.collect_schema().names() if c.startswith(cfg.prefix)]
    if not daioe_cols:
        msg = f"No DAIOE columns found with prefix {cfg.prefix!r}"
        raise ValueError(msg)

    w = pl.col(cfg.weight_col)

    mean_exprs = [pl.col(c).mean().alias(f"{c}_avg") for c in daioe_cols]
    weighted_avg_exprs = [
        pl.when(
            (denom := pl.when(pl.col(c).is_not_null()).then(w).otherwise(None).sum())
            > 0,
        )
        .then((pl.col(c) * w).sum() / denom)
        .otherwise(None)
        .alias(f"{c}_wavg")
        for c in daioe_cols
    ]

    out = (
        lf.group_by(["year", code_col])
        .agg(w.sum().alias("weight_sum"), *mean_exprs, *weighted_avg_exprs)
        .with_columns(pl.lit(level_label).alias("level"))
        .rename({code_col: "ssyk_code"})
    )

    if not cfg.add_percentiles:
        return out

    group_keys = ["year", "level"]
    n_expr = pl.len().over(group_keys)
    rank_expr = (
        pl.col(f"^{cfg.prefix}.*_(avg|wavg)$")
        .rank(method="average", descending=cfg.descending)
        .over(group_keys)
    )

    return out.with_columns(
        (
            pl.when(n_expr > 1).then((rank_expr - 1) / (n_expr - 1)).otherwise(0.0)
            * cfg.pct_scale
        ).name.prefix("pctl_"),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_scb_employment_changes(scb_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate SCB employment by group and compute 1/3/5-year changes."""
    time_col = "year"
    change_keys = [c for c in scb_lf.collect_schema().names() if c != "value"]
    group_keys = [c for c in change_keys if c != time_col]

    return (
        scb_lf.group_by(change_keys)
        .agg(pl.col("value").sum().alias("emp_count"))
        .with_columns(
            pl.col("emp_count")
            .shift(1)
            .over(group_keys, order_by=time_col)
            .alias("_emp_1y"),
            pl.col("emp_count")
            .shift(3)
            .over(group_keys, order_by=time_col)
            .alias("_emp_3y"),
            pl.col("emp_count")
            .shift(5)
            .over(group_keys, order_by=time_col)
            .alias("_emp_5y"),
        )
        .with_columns(
            (pl.col("emp_count") - pl.col("_emp_1y")).alias("chg_1y"),
            (pl.col("emp_count") - pl.col("_emp_3y")).alias("chg_3y"),
            (pl.col("emp_count") - pl.col("_emp_5y")).alias("chg_5y"),
            ((pl.col("emp_count") / pl.col("_emp_1y") - 1) * 100).alias("pct_chg_1y"),
            ((pl.col("emp_count") / pl.col("_emp_3y") - 1) * 100).alias("pct_chg_3y"),
            ((pl.col("emp_count") / pl.col("_emp_5y") - 1) * 100).alias("pct_chg_5y"),
        )
        .drop("_emp_1y", "_emp_3y", "_emp_5y")
    )


def build_daioe_ssyk12(daioe_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Derive SSYK2012 hierarchy codes and filter to SSYK12 publication years."""
    return (
        daioe_lf.with_columns(
            pl.col("ssyk2012_4").str.slice(0, i).alias(f"code_{i}") for i in range(1, 5)
        )
        .drop(pl.col("^ssyk2012.*$"))
        # Filter from 2014: first year of SSYK2012 publication
        .filter(pl.col("year") >= SSYK12_START_YEAR)
    )


def extend_daioe_years(
    daioe_lf: pl.LazyFrame,
    scb_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Extend DAIOE series forward to match latest SCB year.
    Missing years are scaffolded by repeating the last known DAIOE year's
    occupation structure (DAIOE scores frozen at max year).
    """  # noqa: D205
    daioe_max = daioe_lf.select(pl.max("year")).collect().item()
    scb_max = scb_lf.select(pl.max("year")).collect().item()
    missing_years = list(range(daioe_max + 1, scb_max + 1))

    if not missing_years:
        return daioe_lf

    scaffold = (
        daioe_lf.filter(pl.col("year") == daioe_max)
        .drop("year")
        .join(pl.LazyFrame({"year": missing_years}), how="cross")
        .select(daioe_lf.collect_schema().names())
    )

    return pl.concat([daioe_lf, scaffold], how="vertical")


def build_scb_level4_counts(scb_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate SCB employment to SSYK4 level (total across counties and sex)."""
    return (
        scb_lf.filter(pl.col("level") == "SSYK4")
        .group_by(["ssyk_code", "year"])
        .agg(pl.col("value").sum().alias("total_count"))
    )


def build_exposure_levels(
    lf: pl.LazyFrame,
    cfg: AggConfig | None = None,
) -> pl.LazyFrame:
    """Convert weighted percentile ranks (0-100) into 1-5 exposure level columns."""
    if cfg is None:
        cfg = AggConfig()

    pct_cols = [
        c
        for c in lf.collect_schema().names()
        if c.startswith("pctl_daioe_") and c.endswith("_wavg")
    ]

    exposure_exprs = [
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(pl.col(col) <= cfg.pct_l1)
        .then(1)
        .when(pl.col(col) <= cfg.pct_l2)
        .then(2)
        .when(pl.col(col) <= cfg.pct_l3)
        .then(3)
        .when(pl.col(col) <= cfg.pct_l4)
        .then(4)
        .otherwise(5)
        .cast(pl.Int8)
        .alias(f"daioe_{col[len('pctl_daioe_') : -len('_wavg')]}_Level_Exposure")
        for col in pct_cols
    ]

    return lf.with_columns(exposure_exprs)


def main() -> None:
    # --- 1. Load sources lazily ---
    daioe_lf = pl.scan_csv(DAIOE_SOURCE)
    scb_lf = pl.scan_parquet(SCB_SOURCE)

    # --- 2. SCB: employment changes ---
    scb_changes = build_scb_employment_changes(scb_lf)

    # --- 3. DAIOE: derive SSYK2012 hierarchy codes ---
    daioe_ssyk12 = build_daioe_ssyk12(daioe_lf)

    # --- 4. Extend DAIOE years to match SCB coverage ---
    daioe_extended = extend_daioe_years(daioe_ssyk12, scb_lf)

    # --- 5. SCB SSYK4 counts for weighting ---
    scb_level4 = build_scb_level4_counts(scb_lf)

    # Join DAIOE with SCB SSYK4 employment counts (for weighted aggregation)
    daioe_scb = daioe_extended.join(
        scb_level4,
        left_on=["year", "code_4"],
        right_on=["year", "ssyk_code"],
        how="left",
    )

    # --- 6. Aggregate DAIOE metrics across all SSYK levels ---
    levels_map = {
        "code_4": "SSYK4",
        "code_3": "SSYK3",
        "code_2": "SSYK2",
        "code_1": "SSYK1",
    }

    daioe_all_levels = pl.concat(
        [
            aggregate_daioe_level(daioe_scb, col, label)
            for col, label in levels_map.items()
        ],
    ).sort(["level", "year", "ssyk_code"])

    # --- 7. Build 1-5 exposure level columns ---
    daioe_all_levels = build_exposure_levels(daioe_all_levels)

    # --- 8. Final merge: SCB changes + DAIOE exposure ---
    final = scb_changes.join(
        daioe_all_levels,
        on=["year", "ssyk_code"],
        how="left",
    ).drop("level_right")

    # --- Export ---
    final.sink_parquet(OUTPUT_PATH)
    print(f"Exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
