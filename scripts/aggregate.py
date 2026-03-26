import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ======================= Configurations ================================ #


@dataclass(frozen=True)
class Paths:
    """All file paths used by the pipeline."""

    root: Path
    in_file: Path
    map_file: Path
    out_file: Path


def default_paths(root: Path | None = None) -> Paths:
    """Resolve default paths relative to the working directory."""
    root = (root or Path.cwd()).resolve()
    data_dir = root / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    return Paths(
        root=root,
        in_file=data_dir / "scb_yr_regions.parquet",
        map_file=root / "structure_ssyk12.csv",
        out_file=processed / "ssyk12_aggregated_ssyk4_to_ssyk1.parquet",
    )


# ======================= Validation ================================ #


def ensure_inputs(paths: Paths) -> None:
    """Raise if required input files are missing."""
    for path in (paths.in_file, paths.map_file):
        if not path.exists():
            msg = f"Input file not found: {path}"
            raise FileNotFoundError(msg)


# ======================= Loadings ================================ #


def load_scb(paths: Paths) -> pl.DataFrame:
    """Load and normalise the merged SCB parquet."""
    df = pl.read_parquet(paths.in_file)

    required = {"code_4", "county_code", "county", "sex", "year", "value"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    return df.with_columns(
        pl.col("code_4").cast(pl.Utf8),
        pl.col("county_code").cast(pl.Utf8),
        pl.col("county").cast(pl.Utf8),
        pl.col("sex").cast(pl.Utf8),
        pl.col("year").cast(pl.Int64),
        pl.col("value").cast(pl.Int64),
    )


def load_name_map(paths: Paths) -> pl.DataFrame:
    """Load SSYK code → occupation name mapping from CSV."""
    return (
        pl.read_csv(
            paths.map_file,
            schema_overrides={"code": pl.Utf8, "name": pl.Utf8},
        )
        .with_columns(
            pl.col("code").str.strip_chars(),
            pl.col("name").str.strip_chars(),
        )
        .select(["code", "name"])
        .unique(subset=["code"], keep="first")
    )


# ======================= Aggregation ================================ #


def add_ssyk_levels(df: pl.DataFrame) -> pl.LazyFrame:
    """Derive SSYK3/2/1 code columns by slicing code_4."""
    return df.lazy().with_columns(
        ssyk4=pl.col("code_4"),
        ssyk3=pl.col("code_4").str.slice(0, 3),
        ssyk2=pl.col("code_4").str.slice(0, 2),
        ssyk1=pl.col("code_4").str.slice(0, 1),
    )


# group-by dimensions shared across all levels
_GRP_DIMS = ["county_code", "county", "sex", "year"]


def agg_level(lf: pl.LazyFrame, level_col: str, level_name: str) -> pl.LazyFrame:
    """Aggregate value to a single SSYK level, keeping regional/demographic dims."""
    return (
        lf.group_by([level_col, *_GRP_DIMS])
        .agg(pl.col("value").sum().alias("value"))
        .with_columns(pl.lit(level_name).alias("level"))
        .rename({level_col: "ssyk_code"})
        .select(["level", "ssyk_code", *_GRP_DIMS, "value"])
    )


def aggregate_all_levels(lf: pl.LazyFrame) -> pl.DataFrame:
    """Concatenate aggregations for SSYK4 through SSYK1."""
    return pl.concat(
        [
            agg_level(lf, "ssyk4", "SSYK4"),
            agg_level(lf, "ssyk3", "SSYK3"),
            agg_level(lf, "ssyk2", "SSYK2"),
            agg_level(lf, "ssyk1", "SSYK1"),
        ],
        how="vertical",
    ).collect()


# ======================= Name Mapping ================================ #


def map_occupation_names(df: pl.DataFrame, name_map: pl.DataFrame) -> pl.DataFrame:
    """Left-join occupation names onto aggregated data."""
    return (
        df.join(name_map, left_on="ssyk_code", right_on="code", how="left")
        .rename({"name": "occupation"})
        .select(["level", "ssyk_code", "occupation", *_GRP_DIMS, "value"])
    )


# ======================= Diagnostics ================================ #


def log_diagnostics(df: pl.DataFrame) -> None:
    """Log row counts per level and any unmapped SSYK codes."""
    counts = df.group_by("level").agg(pl.len().alias("rows")).sort("level")
    log.info("Rows per level:\n%s", counts)

    unmapped = (
        df.filter(pl.col("occupation").is_null())
        .select(["level", "ssyk_code"])
        .unique()
        .sort(["level", "ssyk_code"])
    )
    if unmapped.height > 0:
        log.warning("Unmapped SSYK codes:\n%s", unmapped)
    else:
        log.info("All SSYK codes mapped successfully.")


# ======================= Entry Point ================================ #


def main() -> None:
    """Orchestrate load, aggregate, name-map, and save."""
    paths = default_paths()
    log.info("in_file  : %s", paths.in_file)
    log.info("map_file : %s", paths.map_file)
    log.info("out_file : %s", paths.out_file)

    ensure_inputs(paths)

    df = load_scb(paths)
    log.info("Loaded %d rows from scb_yr_regions", df.height)

    lf = add_ssyk_levels(df)
    df_agg = aggregate_all_levels(lf)

    name_map = load_name_map(paths)
    df_final = map_occupation_names(df_agg, name_map)

    log_diagnostics(df_final)

    df_final.write_parquet(paths.out_file)
    log.info("Saved → %s  (%d rows)", paths.out_file, df_final.height)


if __name__ == "__main__":
    main()
