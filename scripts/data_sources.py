from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DAIOE_SOURCE: str = (
    "https://raw.githubusercontent.com/joseph-data/07_translate_ssyk/main/"
    "03_translated_files/daioe_ssyk2012_translated.csv"
)

SCB_REMOTE_SOURCE: str = (
    "https://raw.githubusercontent.com/joseph-data/AI_Econ_daioe_years_regions/daioe_pull/"
    "data/processed/ssyk12_aggregated_ssyk4_to_ssyk1.parquet"
)
SCB_LOCAL_SOURCE = PROCESSED_DIR / "ssyk12_aggregated_ssyk4_to_ssyk1.parquet"
SCB_SOURCE: str | Path = (
    SCB_LOCAL_SOURCE if SCB_LOCAL_SOURCE.exists() else SCB_REMOTE_SOURCE
)


def ensure_data_dirs() -> None:
    """Create the repo data directories when they do not exist yet."""
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_scb_source(
    source: str | Path | None = None,
    *,
    prefer_local: bool = True,
) -> str | Path:
    """Return the explicit SCB source or the preferred default."""
    if source is not None:
        return source

    if prefer_local and SCB_LOCAL_SOURCE.exists():
        return SCB_LOCAL_SOURCE

    return SCB_REMOTE_SOURCE


def scan_daioe(source: str | Path | None = None, **scan_kwargs) -> pl.LazyFrame:
    """Scan the translated DAIOE data as a lazy Polars frame."""
    ensure_data_dirs()
    return pl.scan_csv(source or DAIOE_SOURCE, **scan_kwargs)


def scan_scb(
    source: str | Path | None = None,
    *,
    prefer_local: bool = True,
    **scan_kwargs,
) -> pl.LazyFrame:
    """Scan the SCB employment data as a lazy Polars frame."""
    ensure_data_dirs()
    resolved_source = resolve_scb_source(source, prefer_local=prefer_local)
    return pl.scan_parquet(resolved_source, **scan_kwargs)
