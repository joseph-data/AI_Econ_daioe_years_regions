import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pyscbwrapper import SCB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path.cwd().resolve()
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "scb_yr_regions.parquet"
DEDUP_KEYS = ["code_4", "county_code", "sex", "year"]


# =========================== Config Table ================================ #


@dataclass(frozen=True)
class TableSpec:
    """Identifiers and dedup priority for a single SCB table."""

    name: str
    ids: tuple[str, str, str, str, str]
    priority: int  # higher = more recent; wins on overlapping years


TABLE_SPECS = (
    TableSpec("yr_rg_14_to_18", ("en", "AM", "AM0208", "AM0208M", "YREG60"), 1),
    TableSpec("yr_rg_19_to_21", ("en", "AM", "AM0208", "AM0208M", "YREG60N"), 2),
    TableSpec("yr_rg_20_to_24", ("en", "AM", "AM0208", "AM0208M", "YREG60BAS"), 3),
)


# ========================= Variable Helpers ================================ #


def find_variable(variables: list[dict], keyword: str) -> dict:
    keyword_lower = keyword.lower()
    for variable in variables:
        if keyword_lower in variable.get("text", "").lower():
            return variable
    available = [v.get("text") for v in variables]
    msg = f"No variable found containing {keyword!r}. Available: {available}"
    raise ValueError(msg)


def to_query_key(text: str) -> str:
    """Strip whitespace from a variable text to match pyscbwrapper's query key format."""
    return text.replace(" ", "")


def build_lookup(variable: dict) -> dict[str, str]:
    """Map SCB codes to their human-readable labels for a given variable."""
    return dict(zip(variable["values"], variable["valueTexts"], strict=True))


def build_key_positions(
    variables: list[dict],
    observations_code: str,
) -> dict[str, int]:
    keyword_map = {
        "region": "county",
        "occupation": "occupation",
        "sex": "sex",
        "year": "year",
    }
    positions: dict[str, int] = {}
    key_index = 0

    for variable in variables:
        if variable["code"] == observations_code:
            continue
        text = variable.get("text", "").lower()
        for keyword, field in keyword_map.items():
            if keyword in text:
                positions[field] = key_index
                break
        key_index += 1

    missing = {"county", "occupation", "sex", "year"} - positions.keys()
    if missing:
        msg = f"Missing key positions for: {', '.join(sorted(missing))}"
        raise ValueError(msg)

    return positions


# ======================= SCB Fetch ================================ #


def fetch_table(spec: TableSpec) -> pl.DataFrame:
    """
    Fetch and transform a single SCB table into a Polars DataFrame.

    Queries only county-level regions and the first two sexes (men/women).
    Appends _source_table and _priority columns used during deduplication.
    """
    log.info("Fetching %s", spec.name)
    scb = SCB(*spec.ids)
    variables: list[dict] = scb.info()["variables"]

    occupation_var = find_variable(variables, "occupation")
    sex_var = find_variable(variables, "sex")
    region_var = find_variable(variables, "region")
    observations_var = find_variable(variables, "observations")

    # filter to county-level regions only (excludes municipalities, national totals)
    county_pairs = [
        (code, label)
        for code, label in zip(
            region_var["values"],
            region_var["valueTexts"],
            strict=True,
        )
        if label.lower().endswith("county")
    ]
    if not county_pairs:
        msg = f"{spec.name}: no county regions found"
        raise RuntimeError(msg)

    county_codes, county_labels = zip(*county_pairs, strict=True)

    scb.set_query(
        **{
            to_query_key(occupation_var["text"]): occupation_var["valueTexts"],
            to_query_key(sex_var["text"]): sex_var["valueTexts"][:2],
            to_query_key(region_var["text"]): list(county_labels),
            to_query_key(observations_var["text"]): observations_var["valueTexts"][0],
        },
    )

    response = scb.get_data()
    raw_rows: list[dict] | None = response.get("data")
    if not raw_rows:
        msg = f"Empty or missing data in SCB response for {spec.name}"
        raise RuntimeError(msg)

    # resolve key positions dynamically in case column order varies across tables
    positions = build_key_positions(variables, observations_var["code"])

    df = (
        pl.DataFrame(raw_rows)
        .with_columns(
            county_code=pl.col("key").list.get(positions["county"]),
            code_4=pl.col("key").list.get(positions["occupation"]),
            sex_code=pl.col("key").list.get(positions["sex"]),
            year=pl.col("key").list.get(positions["year"]),
            value=pl.col("values")
            .list.get(0)
            .cast(pl.Int64, strict=False),  # ".." → null
        )
        .with_columns(
            county=pl.col("county_code").replace(
                dict(zip(county_codes, county_labels, strict=True)),
            ),
            occupation=pl.col("code_4").replace(build_lookup(occupation_var)),
            sex=pl.col("sex_code").replace(build_lookup(sex_var)),
        )
        # drop aggregate codes (total and unspecified occupations)
        .filter(~pl.col("code_4").is_in(["0002", "0000"]))
        .select(
            ["code_4", "occupation", "county_code", "county", "sex", "year", "value"],
        )
        .with_columns(
            _source_table=pl.lit(spec.name),
            _priority=pl.lit(spec.priority),
        )
    )

    log.info("%s: %d rows", spec.name, df.height)
    return df


# ======================= Threaded Fetch ================================ #


def fetch_all_tables() -> dict[str, pl.DataFrame]:
    """Fetch all SCB tables concurrently and return results keyed by table name."""
    results: dict[str, pl.DataFrame] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(TABLE_SPECS),
    ) as executor:
        future_to_spec = {
            executor.submit(fetch_table, spec): spec for spec in TABLE_SPECS
        }
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                results[spec.name] = future.result()
            except Exception as exc:
                log.exception("%s — failed", spec.name)
                raise

    return results


# ======================= Overlap Diagnostics ================================ #


def collect_year_overlaps(results: dict[str, pl.DataFrame]) -> list[str]:
    """Return human-readable strings describing year overlaps between table pairs."""
    years_by_table = {
        spec.name: set(results[spec.name]["year"].unique().to_list())
        for spec in TABLE_SPECS
    }
    overlaps = []
    for i, left in enumerate(TABLE_SPECS):
        for right in TABLE_SPECS[i + 1 :]:
            shared = sorted(years_by_table[left.name] & years_by_table[right.name])
            if shared:
                overlaps.append(f"{left.name} vs {right.name}: {', '.join(shared)}")
    return overlaps


# ======================= Merge & Deduplicate ================================ #


def combine_tables(results: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, int]:
    """
    Concatenate all tables and deduplicate on DEDUP_KEYS.

    Rows from higher-priority (more recent) tables are kept when years overlap.
    Returns the cleaned DataFrame and the number of duplicate rows removed.
    """
    combined = pl.concat([results[spec.name] for spec in TABLE_SPECS], how="vertical")
    before = combined.height

    deduped = (
        combined.sort("_priority", descending=True)
        .unique(subset=DEDUP_KEYS, keep="first")
        .drop(["_source_table", "_priority"])
        .sort(["code_4", "county_code", "sex", "year"])
    )

    return deduped, before - deduped.height


# ======================= Entry Point ================================ #


def main() -> None:
    """Orchestrate fetch, merge, diagnostics, and save."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = fetch_all_tables()
    df, duplicate_count = combine_tables(results)

    for overlap in collect_year_overlaps(results):
        log.info("Year overlap: %s", overlap)

    log.info("Duplicate rows removed: %d", duplicate_count)
    log.info("Final shape: %s", df.shape)

    df.write_parquet(OUTPUT_PATH)
    log.info("Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
