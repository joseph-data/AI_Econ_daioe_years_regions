import html
import json
from functools import cache
from pathlib import Path

import polars as pl
from shiny import reactive
from shiny.express import input, render, ui

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "daioe_scb_years_all_levels.parquet"

DATA = pl.read_parquet(DATA_PATH)
YEARS = sorted(DATA.get_column("year").unique().to_list())
FIRST_YEAR = YEARS[0]
LATEST_YEAR = YEARS[-1]

LEVEL_OPTIONS = {
    "SSYK1": "SSYK 1",
    "SSYK2": "SSYK 2",
    "SSYK3": "SSYK 3",
    "SSYK4": "SSYK 4",
}

METRIC_OPTIONS = {
    "daioe_genai_wavg": "Generative AI exposure",
    "daioe_allapps_wavg": "All applications exposure",
    "daioe_lngmod_wavg": "Language model exposure",
    "daioe_readcompr_wavg": "Reading comprehension exposure",
    "daioe_translat_wavg": "Translation exposure",
    "daioe_speechrec_wavg": "Speech recognition exposure",
    "daioe_imggen_wavg": "Image generation exposure",
    "daioe_imgrec_wavg": "Image recognition exposure",
    "daioe_imgcompr_wavg": "Image comprehension exposure",
    "daioe_videogames_wavg": "Video game exposure",
    "daioe_stratgames_wavg": "Strategy game exposure",
}

COUNTY_OPTIONS = {
    row["county_code"]: row["county"].removesuffix(" county")
    for row in DATA.select(["county_code", "county"])
    .unique()
    .sort("county_code")
    .iter_rows(named=True)
}
DEFAULT_FOCUS_COUNTY = "01" if "01" in COUNTY_OPTIONS else next(iter(COUNTY_OPTIONS))
DEFAULT_COMPARE_COUNTY = (
    "14"
    if "14" in COUNTY_OPTIONS and DEFAULT_FOCUS_COUNTY != "14"
    else next(code for code in COUNTY_OPTIONS if code != DEFAULT_FOCUS_COUNTY)
)

MAP_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="/vendor/leaflet.css" />
  <style>
    :root {
      --ink: #0c0a3e;
      --ink-soft: #20366d;
      --paper: rgba(255, 255, 255, 0.95);
      --line: rgba(12, 10, 62, 0.12);
      --gold-1: #f5ead2;
      --gold-2: #efc97d;
      --gold-3: #d99a3d;
      --gold-4: #ba6a27;
      --gold-5: #7f3f19;
      --muted: #5e6887;
      --missing: #d9dce7;
      --focus: #ba274a;
    }

    html, body, #map {
      height: 100%;
      margin: 0;
    }

    body {
      background:
        radial-gradient(circle at top right, rgba(77, 108, 250, 0.12), transparent 34%),
        linear-gradient(180deg, #eef3ff 0%, #f8f7f4 56%, #ffffff 100%);
      color: var(--ink);
      font-family: "Nunito Sans", "Segoe UI", sans-serif;
    }

    #map,
    .leaflet-container {
      background:
        radial-gradient(circle at top, rgba(77, 108, 250, 0.08), transparent 42%),
        linear-gradient(180deg, #e9eefc 0%, #f7f6f2 100%);
    }

    .map-banner,
    .info-panel,
    .legend {
      backdrop-filter: blur(12px);
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 16px 38px rgba(12, 10, 62, 0.12);
      color: var(--ink);
    }

    .map-banner {
      padding: 12px 14px;
      min-width: 220px;
    }

    .map-banner .eyebrow,
    .info-panel .eyebrow {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
      text-transform: uppercase;
    }

    .map-banner h3,
    .info-panel h4 {
      font-size: 17px;
      line-height: 1.1;
      margin: 0;
    }

    .map-banner .sub,
    .info-panel .sub,
    .legend .sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }

    .info-panel,
    .legend {
      padding: 12px 14px;
    }

    .info-panel .metric {
      font-size: 13px;
      margin-top: 8px;
    }

    .info-panel .metric strong {
      font-size: 16px;
    }

    .legend {
      min-width: 220px;
    }

    .legend-bar {
      height: 12px;
      margin: 10px 0 8px;
      border-radius: 999px;
      border: 1px solid rgba(12, 10, 62, 0.1);
      background: linear-gradient(
        90deg,
        var(--gold-1) 0%,
        var(--gold-2) 24%,
        var(--gold-3) 50%,
        var(--gold-4) 76%,
        var(--gold-5) 100%
      );
    }

    .legend-scale {
      display: flex;
      font-size: 11px;
      gap: 10px;
      justify-content: space-between;
      color: var(--muted);
    }

    .legend-note {
      align-items: center;
      color: var(--muted);
      display: flex;
      font-size: 11px;
      gap: 8px;
      margin-top: 10px;
    }

    .legend-swatch {
      border: 2px dashed var(--focus);
      border-radius: 999px;
      display: inline-block;
      height: 10px;
      width: 18px;
    }

    .muted {
      color: var(--muted);
    }

    .leaflet-tooltip {
      background: rgba(12, 10, 62, 0.94);
      border: 0;
      border-radius: 14px;
      box-shadow: 0 18px 32px rgba(12, 10, 62, 0.18);
      color: white;
      padding: 10px 12px;
    }

    .leaflet-tooltip .muted {
      color: rgba(255, 255, 255, 0.74);
    }
  </style>
</head>
<body>
  <div id="map"></div>

  <script src="/vendor/leaflet.js"></script>
  <script>
    const payload = __PAYLOAD__;
    const colors = ["#f5ead2", "#efc97d", "#d99a3d", "#ba6a27", "#7f3f19"];
    const missingColor = "#d9dce7";
    const focusColor = "#ba274a";

    function fmtNumber(value, digits = 3) {
      return new Intl.NumberFormat("en-US", {
        minimumFractionDigits: 0,
        maximumFractionDigits: digits,
      }).format(value);
    }

    function fmtInt(value) {
      return new Intl.NumberFormat("en-US", {
        maximumFractionDigits: 0,
      }).format(value);
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function pickColor(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return missingColor;
      }

      if (payload.min === payload.max) {
        return colors[Math.floor(colors.length / 2)];
      }

      const share = clamp((value - payload.min) / (payload.max - payload.min), 0, 0.999999);
      const index = Math.min(colors.length - 1, Math.floor(share * colors.length));
      return colors[index];
    }

    function defaultInfoHtml() {
      const focusItem = payload.data[payload.focus_code];
      if (!focusItem) {
        return `
          <div class="eyebrow">Regional snapshot</div>
          <h4>${payload.metric_label}</h4>
          <div class="sub">${payload.level_label} in ${payload.year}</div>
        `;
      }

      return `
        <div class="eyebrow">County spotlight</div>
        <h4>${focusItem.county}</h4>
        <div class="metric">${payload.metric_label}: <strong>${fmtNumber(focusItem.value)}</strong></div>
        <div class="sub">Employment: ${fmtInt(focusItem.employment)}</div>
        <div class="sub">${fmtNumber(focusItem.value - payload.national_value)} vs national average</div>
      `;
    }

    const map = L.map("map", {
      attributionControl: false,
      zoomControl: true,
      minZoom: 4.4,
      scrollWheelZoom: true,
    });

    const banner = L.control({ position: "topleft" });
    banner.onAdd = function () {
      const div = L.DomUtil.create("div", "map-banner");
      div.innerHTML = `
        <div class="eyebrow">Sweden county map</div>
        <h3>${payload.metric_label}</h3>
        <div class="sub">${payload.level_label} • ${payload.year}</div>
      `;
      return div;
    };
    banner.addTo(map);

    const info = L.control({ position: "topright" });
    info.onAdd = function () {
      this._div = L.DomUtil.create("div", "info-panel");
      this.update();
      return this._div;
    };
    info.update = function (item, municipality) {
      if (!item) {
        this._div.innerHTML = defaultInfoHtml();
        return;
      }

      this._div.innerHTML = `
        <div class="eyebrow">Hovered area</div>
        <h4>${item.county}</h4>
        <div class="metric">${payload.metric_label}: <strong>${fmtNumber(item.value)}</strong></div>
        <div class="sub">Employment: ${fmtInt(item.employment)}</div>
        <div class="sub">${municipality}</div>
        <div class="sub">County code ${item.county_code}</div>
      `;
    };
    info.addTo(map);

    const legend = L.control({ position: "bottomright" });
    legend.onAdd = function () {
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML = `
        <div class="eyebrow">Color scale</div>
        <div><strong>${payload.metric_label}</strong></div>
        <div class="legend-bar"></div>
        <div class="legend-scale">
          <span>${fmtNumber(payload.min)}</span>
          <span>${fmtNumber((payload.min + payload.max) / 2)}</span>
          <span>${fmtNumber(payload.max)}</span>
        </div>
        <div class="legend-note">
          <span class="legend-swatch"></span>
          <span>Selected county outlined</span>
        </div>
        <div class="sub">Scale is fixed for the selected occupation level across all years.</div>
      `;
      return div;
    };
    legend.addTo(map);

    let geoLayer = null;

    fetch("/swedish_municipalities.geojson")
      .then((response) => response.json())
      .then((geojson) => {
        geoLayer = L.geoJSON(geojson, {
          style: (feature) => {
            const countyCode = String(feature.properties.k_id).slice(0, 2);
            const item = payload.data[countyCode];
            const isFocus = countyCode === payload.focus_code;

            return {
              color: isFocus ? focusColor : "rgba(255, 255, 255, 0.82)",
              dashArray: isFocus ? "3 3" : null,
              fillColor: item ? pickColor(item.value) : missingColor,
              fillOpacity: item ? (isFocus ? 0.96 : 0.88) : 0.36,
              weight: isFocus ? 1.7 : 0.8,
            };
          },
          onEachFeature: (feature, layer) => {
            const countyCode = String(feature.properties.k_id).slice(0, 2);
            const item = payload.data[countyCode];
            const municipality = feature.properties.name;
            const tooltipHtml = item
              ? `<strong>${item.county}</strong><br>${payload.metric_label}: ${fmtNumber(item.value)}<br>Employment: ${fmtInt(item.employment)}<br><span class="muted">${municipality}</span>`
              : `<strong>${municipality}</strong><br>No county data available`;

            layer.bindTooltip(tooltipHtml, {
              direction: "auto",
              sticky: true,
            });

            layer.on({
              click: (event) => {
                map.fitBounds(event.target.getBounds(), {
                  maxZoom: 8,
                  padding: [18, 18],
                });
                info.update(item, municipality);
              },
              mouseover: (event) => {
                const isFocus = countyCode === payload.focus_code;
                event.target.setStyle({
                  color: isFocus ? focusColor : "#0c0a3e",
                  fillOpacity: 1,
                  weight: isFocus ? 2.1 : 1.35,
                });

                if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
                  event.target.bringToFront();
                }

                info.update(item, municipality);
              },
              mouseout: (event) => {
                geoLayer.resetStyle(event.target);
                info.update();
              },
            });
          },
        }).addTo(map);

        map.fitBounds(geoLayer.getBounds(), {
          padding: [18, 18],
        });
      })
      .catch((error) => {
        info._div.innerHTML = `
          <div class="eyebrow">Map failed to load</div>
          <h4>We could not render the geographic layer</h4>
          <div class="sub">${String(error)}</div>
        `;
      });
  </script>
</body>
</html>
"""


def current_year() -> int:
    return int(input.year() or LATEST_YEAR)


def current_level() -> str:
    return input.level() or "SSYK4"


def current_metric() -> str:
    return input.metric() or "daioe_genai_wavg"


def focus_county_code() -> str:
    code = input.focus_county()
    return code if code in COUNTY_OPTIONS else DEFAULT_FOCUS_COUNTY


def compare_county_code() -> str:
    code = input.compare_county()
    return code if code in COUNTY_OPTIONS else DEFAULT_COMPARE_COUNTY


def short_county_name(name: str) -> str:
    return name.removesuffix(" county")


def format_value(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".")


def format_count(value: int) -> str:
    return f"{value:,}"


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def format_signed_value(value: float, digits: int = 3) -> str:
    prefix = "+" if value > 0 else ""
    return f"{prefix}{format_value(value, digits)}"


def delta_direction(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def describe_rank_change(value: int) -> str:
    if value > 0:
        return f"Up {value} places since {FIRST_YEAR}"
    if value < 0:
        return f"Down {abs(value)} places since {FIRST_YEAR}"
    if current_year() == FIRST_YEAR:
        return f"Baseline year {FIRST_YEAR}"
    return f"Same rank as {FIRST_YEAR}"


def make_stat_card(label: str, value: str, detail: str, tone: str = "ink"):
    return ui.tags.div(
        ui.tags.div(label, class_="metric-card-label"),
        ui.tags.div(value, class_="metric-card-value"),
        ui.tags.div(detail, class_="metric-card-detail"),
        class_=f"metric-card tone-{tone}",
    )


def make_delta_pill(text: str, direction: str):
    return ui.tags.span(text, class_=f"delta-pill {direction}")


def build_map_srcdoc(payload: dict) -> str:
    return MAP_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))


@cache
def county_summary(level: str, year: int, metric: str) -> pl.DataFrame:
    frame = DATA.filter((pl.col("level") == level) & (pl.col("year") == year))
    return (
        frame.group_by(["county_code", "county"])
        .agg(
            pl.col("emp_count").sum().alias("employment"),
            (
                (pl.col(metric) * pl.col("emp_count")).sum() / pl.col("emp_count").sum()
            ).alias("value"),
        )
        .sort("value", descending=True)
    )


@cache
def county_timeseries(level: str, metric: str) -> pl.DataFrame:
    frame = (
        DATA.filter(pl.col("level") == level)
        .group_by(["year", "county_code", "county"])
        .agg(
            pl.col("emp_count").sum().alias("employment"),
            (
                (pl.col(metric) * pl.col("emp_count")).sum() / pl.col("emp_count").sum()
            ).alias("value"),
        )
        .with_columns(
            pl.col("value")
            .rank("ordinal", descending=True)
            .over("year")
            .cast(pl.Int64)
            .alias("rank"),
        )
        .sort(["year", "rank"])
    )
    return frame


@cache
def national_timeseries(level: str, metric: str) -> pl.DataFrame:
    return (
        DATA.filter(pl.col("level") == level)
        .group_by("year")
        .agg(
            pl.col("emp_count").sum().alias("employment"),
            (
                (pl.col(metric) * pl.col("emp_count")).sum() / pl.col("emp_count").sum()
            ).alias("value"),
        )
        .sort("year")
    )


@cache
def county_metric_bounds(level: str, metric: str) -> tuple[float, float]:
    stats = county_timeseries(level, metric).select(
        pl.col("value").min().alias("min_value"),
        pl.col("value").max().alias("max_value"),
    ).row(0, named=True)
    return float(stats["min_value"]), float(stats["max_value"])


@cache
def occupation_profile(level: str, year: int, county_code: str, metric: str) -> pl.DataFrame:
    frame = (
        DATA.filter(
            (pl.col("level") == level)
            & (pl.col("year") == year)
            & (pl.col("county_code") == county_code),
        )
        .group_by("occupation")
        .agg(
            pl.col("emp_count").sum().alias("employment"),
            (
                (pl.col(metric) * pl.col("emp_count")).sum() / pl.col("emp_count").sum()
            ).alias("value"),
        )
        .drop_nulls("value")
        .sort("employment", descending=True)
    )

    if frame.height == 0:
        return frame

    total_employment = float(frame.get_column("employment").sum())
    county_value = float(
        frame.select(
            ((pl.col("value") * pl.col("employment")).sum() / pl.col("employment").sum()).alias(
                "county_value",
            ),
        ).item(),
    )

    return frame.with_columns(
        (pl.col("employment") / total_employment).alias("employment_share"),
        (pl.col("value") * pl.col("employment") / total_employment).alias(
            "score_contribution",
        ),
        (pl.col("value") / county_value).alias("relative_to_county_score"),
    )


def county_profile_for(
    code: str, summary: pl.DataFrame, trend: pl.DataFrame, rank_shift: pl.DataFrame, year: int,
) -> dict:
    summary_row = summary.filter(pl.col("county_code") == code).row(0, named=True)
    trend_row = trend.filter(pl.col("county_code") == code).sort("year")
    shift_row = rank_shift.filter(pl.col("county_code") == code).row(0, named=True)
    previous_row = trend_row.filter(pl.col("year") < year).tail(1)

    return {
        "code": code,
        "county": summary_row["county"],
        "employment": int(summary_row["employment"]),
        "rank": int(summary_row["rank"]),
        "rank_change": int(shift_row["rank_change"]),
        "value": float(summary_row["value"]),
        "value_change": float(shift_row["value_change"]),
        "vs_national": float(summary_row["vs_national"]),
        "yoy_change": None
        if previous_row.height == 0
        else float(summary_row["value"] - previous_row.row(0, named=True)["value"]),
    }


def make_county_table(frame: pl.DataFrame, highlight_code: str | None = None):
    if frame.height == 0:
        return ui.tags.div("No county data available for this selection.", class_="empty-state")

    max_value = float(frame.get_column("value").max()) or 1.0
    rows = []

    for row in frame.iter_rows(named=True):
        bar_width = max(8.0, (float(row["value"]) / max_value) * 100)
        delta = float(row["vs_national"])
        delta_class = delta_direction(delta)
        row_class = "table-highlight" if row["county_code"] == highlight_code else ""

        county_cell = [ui.tags.div(short_county_name(row["county"]), class_="county-name")]
        if row["county_code"] == highlight_code:
            county_cell.append(ui.tags.span("Focus", class_="row-chip"))

        rows.append(
            ui.tags.tr(
                ui.tags.td(f"#{int(row['rank'])}", class_="rank-col"),
                ui.tags.td(*county_cell, class_="county-col"),
                ui.tags.td(
                    ui.tags.div(format_value(float(row["value"])), class_="table-metric-value"),
                    ui.tags.div(
                        ui.tags.div(
                            class_="table-metric-fill", style=f"width: {bar_width:.1f}%",
                        ),
                        class_="table-metric-track",
                    ),
                    class_="table-metric-cell",
                ),
                ui.tags.td(
                    make_delta_pill(format_signed_value(delta), delta_class),
                    class_="delta-col",
                ),
                ui.tags.td(format_count(int(row["employment"])), class_="employment-col"),
                class_=row_class,
            ),
        )

    return ui.tags.div(
        ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Rank"),
                    ui.tags.th("County"),
                    ui.tags.th("Exposure"),
                    ui.tags.th("Vs national"),
                    ui.tags.th("Employment"),
                ),
            ),
            ui.tags.tbody(*rows),
            class_="dashboard-table",
        ),
        class_="table-wrap",
    )


def make_bar_list(
    frame: pl.DataFrame,
    *,
    label_col: str,
    bar_col: str,
    value_builder,
    meta_builder,
    limit: int = 8,
):
    if frame.height == 0:
        return ui.tags.div("No occupation detail is available for this county.", class_="empty-state")

    display = frame.head(limit)
    max_bar = float(display.get_column(bar_col).max()) or 1.0
    rows = []

    for row in display.iter_rows(named=True):
        bar_width = max(8.0, (float(row[bar_col]) / max_bar) * 100)
        rows.append(
            ui.tags.div(
                ui.tags.div(
                    ui.tags.div(row[label_col], class_="bar-row-label"),
                    ui.tags.div(value_builder(row), class_="bar-row-value"),
                    class_="bar-row-top",
                ),
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.div(class_="bar-fill", style=f"width: {bar_width:.1f}%"),
                        class_="bar-track",
                    ),
                    ui.tags.div(meta_builder(row), class_="bar-row-meta"),
                    class_="bar-row-bottom",
                ),
                class_="bar-row",
            ),
        )

    return ui.tags.div(*rows, class_="bar-list")


def build_trend_svg(series: list[dict]) -> str:
    all_points = [point for item in series for point in item["points"]]
    if not all_points:
        return "<div class='empty-state'>No time series available.</div>"

    years = sorted({int(point["year"]) for point in all_points})
    values = [float(point["value"]) for point in all_points]

    width = 860
    height = 320
    left = 56
    right = 18
    top = 16
    bottom = 44

    min_value = min(values)
    max_value = max(values)
    spread = max_value - min_value
    padding = spread * 0.16 if spread else max(abs(max_value) * 0.16, 0.12)
    lo = min_value - padding
    hi = max_value + padding

    def x_pos(year: int) -> float:
        if len(years) == 1:
            return (width - left - right) / 2 + left
        return left + ((year - years[0]) / (years[-1] - years[0])) * (width - left - right)

    def y_pos(value: float) -> float:
        return height - bottom - ((value - lo) / (hi - lo)) * (height - top - bottom)

    grid = []
    for index in range(5):
        tick = lo + ((hi - lo) / 4) * index
        y = y_pos(tick)
        grid.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' "
            "stroke='rgba(12,10,62,0.10)' stroke-width='1' />",
        )
        grid.append(
            f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' "
            "font-size='11' fill='#5e6887'>"
            f"{html.escape(format_value(tick, 2))}</text>",
        )

    x_labels = []
    for year in years:
        x = x_pos(year)
        x_labels.append(
            f"<text x='{x:.2f}' y='{height - 16}' text-anchor='middle' "
            "font-size='11' fill='#5e6887'>"
            f"{year}</text>",
        )

    lines = []
    for item in series:
        points = [
            (x_pos(int(point["year"])), y_pos(float(point["value"])))
            for point in item["points"]
        ]
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash = " stroke-dasharray='8 6'" if item.get("dash") else ""
        lines.append(
            f"<polyline fill='none' points='{polyline}' stroke='{item['color']}' "
            f"stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round'{dash} />",
        )
        lines.extend(
            [
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.3' fill='{item['color']}' "
                "stroke='white' stroke-width='2' />"
                for x, y in points
            ],
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='County exposure trend chart'>"
        f"{''.join(grid)}"
        f"{''.join(lines)}"
        f"{''.join(x_labels)}"
        "</svg>"
    )


def make_chart_legend(series: list[dict]):
    items = []
    for item in series:
        swatch_class = "chart-legend-line dashed" if item.get("dash") else "chart-legend-line"
        items.append(
            ui.tags.div(
                ui.tags.span(
                    class_=swatch_class,
                    style=f"--legend-color: {item['color']}",
                ),
                ui.tags.span(item["name"], class_="chart-legend-label"),
                class_="chart-legend-item",
            ),
        )

    return ui.tags.div(*items, class_="chart-legend")


ui.page_opts(
    title="AI Exposure Atlas for Swedish Counties",
    theme=ui.Theme.from_brand(__file__),
    fillable=True,
    lang="en",
    full_width=True,
)

ui.tags.style(
    """
    :root {
      --ink: #0c0a3e;
      --ink-soft: #1d3163;
      --sand: #f9f7f1;
      --paper: rgba(255, 255, 255, 0.92);
      --paper-strong: #ffffff;
      --line: rgba(12, 10, 62, 0.08);
      --blue-soft: rgba(77, 108, 250, 0.10);
      --gold: #c78a2c;
      --gold-soft: rgba(199, 138, 44, 0.14);
      --coral: #ba274a;
      --coral-soft: rgba(186, 39, 74, 0.12);
      --slate: #5e6887;
      --shadow: 0 22px 52px rgba(12, 10, 62, 0.08);
      --radius-lg: 28px;
      --radius-md: 22px;
    }

    html,
    body {
      background:
        radial-gradient(circle at top right, rgba(77, 108, 250, 0.12), transparent 28%),
        linear-gradient(180deg, #f2f5fd 0%, #f8f7f4 44%, #ffffff 100%);
      color: var(--ink);
    }

    .bslib-sidebar-layout {
      gap: 1.25rem;
      padding: 1.15rem;
    }

    .control-sidebar.sidebar {
      background:
        radial-gradient(circle at top right, rgba(199, 138, 44, 0.16), transparent 36%),
        linear-gradient(180deg, #0c0a3e 0%, #1b2d5f 100%);
      border: 0;
      border-radius: var(--radius-lg);
      box-shadow: 0 26px 54px rgba(12, 10, 62, 0.2);
      color: #f5f7ff;
    }

    .control-sidebar .sidebar-title {
      color: white;
      font-family: "Montserrat", sans-serif;
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.1;
      margin-bottom: 0.85rem;
    }

    .sidebar-kicker {
      color: rgba(255, 255, 255, 0.72);
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
    }

    .sidebar-copy {
      color: rgba(255, 255, 255, 0.78);
      font-size: 0.95rem;
      line-height: 1.55;
      margin: 0 0 1rem;
    }

    .control-sidebar .form-label,
    .control-sidebar .control-label {
      color: rgba(235, 240, 255, 0.9);
      font-size: 0.79rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 0.35rem;
      text-transform: uppercase;
    }

    .control-sidebar .form-control,
    .control-sidebar .form-select,
    .control-sidebar .selectize-input,
    .control-sidebar .irs--shiny .irs-bar,
    .control-sidebar .irs--shiny .irs-single {
      border-radius: 14px;
    }

    .control-sidebar .form-control,
    .control-sidebar .form-select,
    .control-sidebar .selectize-input {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      color: white;
    }

    .control-sidebar .selectize-input > div,
    .control-sidebar .selectize-input input {
      color: white;
    }

    .control-sidebar .irs--shiny .irs-line {
      background: rgba(255, 255, 255, 0.14);
      border: 0;
    }

    .control-sidebar .irs--shiny .irs-bar {
      background: linear-gradient(90deg, #4d6cfa 0%, #c78a2c 100%);
      border: 0;
    }

    .control-sidebar .irs--shiny .irs-handle {
      background: white;
      border: 2px solid rgba(255, 255, 255, 0.32);
      box-shadow: none;
      top: 21px;
    }

    .control-sidebar .irs--shiny .irs-min,
    .control-sidebar .irs--shiny .irs-max,
    .control-sidebar .irs--shiny .irs-from,
    .control-sidebar .irs--shiny .irs-to,
    .control-sidebar .irs--shiny .irs-single {
      background: rgba(255, 255, 255, 0.12);
      color: white;
    }

    .sidebar-note {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 18px;
      margin-top: 1rem;
      padding: 0.95rem 1rem;
    }

    .sidebar-note-title {
      color: white;
      font-size: 0.88rem;
      font-weight: 700;
      margin-bottom: 0.3rem;
    }

    .sidebar-note-copy {
      color: rgba(255, 255, 255, 0.76);
      font-size: 0.88rem;
      line-height: 1.5;
      margin: 0;
    }

    .hero-banner {
      background:
        radial-gradient(circle at top right, rgba(199, 138, 44, 0.18), transparent 28%),
        linear-gradient(135deg, rgba(12, 10, 62, 0.98) 0%, rgba(31, 49, 101, 0.96) 52%, rgba(35, 58, 118, 0.94) 100%);
      border-radius: var(--radius-lg);
      box-shadow: 0 28px 58px rgba(12, 10, 62, 0.18);
      color: white;
      margin-bottom: 1.2rem;
      overflow: hidden;
      padding: 1.6rem 1.7rem;
      position: relative;
    }

    .hero-banner::after {
      background: radial-gradient(circle, rgba(186, 39, 74, 0.28) 0%, transparent 68%);
      content: "";
      height: 16rem;
      pointer-events: none;
      position: absolute;
      right: -3rem;
      top: -5rem;
      width: 16rem;
    }

    .hero-eyebrow {
      color: rgba(255, 255, 255, 0.72);
      font-size: 0.77rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      margin-bottom: 0.45rem;
      text-transform: uppercase;
    }

    .hero-title {
      color: white;
      font-family: "Montserrat", sans-serif;
      font-size: clamp(1.9rem, 3vw, 2.85rem);
      line-height: 1.02;
      margin: 0;
      max-width: 13ch;
    }

    .hero-copy {
      color: rgba(255, 255, 255, 0.82);
      font-size: 1rem;
      line-height: 1.65;
      margin: 0.8rem 0 1rem;
      max-width: 70ch;
      position: relative;
      z-index: 1;
    }

    .hero-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem;
      margin-bottom: 1rem;
      position: relative;
      z-index: 1;
    }

    .hero-chip {
      backdrop-filter: blur(10px);
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 999px;
      color: white;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.42rem 0.8rem;
    }

    .hero-highlight-grid {
      display: grid;
      gap: 0.9rem;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      position: relative;
      z-index: 1;
    }

    .hero-mini-card {
      backdrop-filter: blur(10px);
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 18px;
      padding: 0.95rem 1rem;
    }

    .hero-mini-label {
      color: rgba(255, 255, 255, 0.66);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      margin-bottom: 0.35rem;
      text-transform: uppercase;
    }

    .hero-mini-value {
      color: white;
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.05;
    }

    .hero-mini-copy {
      color: rgba(255, 255, 255, 0.72);
      font-size: 0.88rem;
      margin-top: 0.35rem;
    }

    .nav-tabs {
      border-bottom: 1px solid var(--line);
      gap: 0.3rem;
      margin-bottom: 0.8rem;
    }

    .nav-tabs .nav-link {
      background: transparent;
      border: 0;
      border-radius: 999px;
      color: var(--slate);
      font-weight: 700;
      padding: 0.65rem 1rem;
    }

    .nav-tabs .nav-link.active {
      background: rgba(12, 10, 62, 0.08);
      color: var(--ink);
    }

    .card-header {
      background: transparent;
      border-bottom: 1px solid var(--line);
    }

    .tab-shell-title {
      color: var(--ink);
      font-family: "Montserrat", sans-serif;
      font-size: 1rem;
      font-weight: 700;
    }

    .section-copy {
      color: var(--slate);
      font-size: 0.96rem;
      margin: 0 0 1rem;
    }

    .metrics-grid {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 1rem;
    }

    .metric-card {
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      box-shadow: var(--shadow);
      overflow: hidden;
      padding: 1rem 1.05rem 1.1rem;
      position: relative;
    }

    .metric-card::before {
      background: var(--tone);
      content: "";
      height: 4px;
      inset: 0 0 auto;
      position: absolute;
    }

    .tone-ink {
      --tone: var(--ink);
    }

    .tone-gold {
      --tone: var(--gold);
    }

    .tone-coral {
      --tone: var(--coral);
    }

    .tone-blue {
      --tone: #4d6cfa;
    }

    .metric-card-label {
      color: var(--slate);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .metric-card-value {
      color: var(--ink);
      font-size: 1.55rem;
      font-weight: 700;
      line-height: 1.05;
      margin-top: 0.45rem;
    }

    .metric-card-detail {
      color: var(--slate);
      font-size: 0.93rem;
      line-height: 1.5;
      margin-top: 0.4rem;
    }

    .app-card {
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .card-heading {
      color: var(--ink);
      font-family: "Montserrat", sans-serif;
      font-size: 1.18rem;
      font-weight: 700;
      margin: 0;
    }

    .card-copy {
      color: var(--slate);
      font-size: 0.94rem;
      line-height: 1.55;
      margin: 0.45rem 0 0;
    }

    .map-frame {
      background: #eef2ff;
      border: 0;
      border-radius: 20px;
      min-height: 720px;
      width: 100%;
    }

    .focus-panel {
      background:
        radial-gradient(circle at top right, rgba(77, 108, 250, 0.08), transparent 38%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(246, 247, 251, 0.96) 100%);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 1rem 1rem 0.95rem;
    }

    .focus-panel-copy {
      color: var(--slate);
      font-size: 0.95rem;
      line-height: 1.6;
      margin: 0 0 0.8rem;
    }

    .focus-grid {
      display: grid;
      gap: 0.7rem;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .focus-stat {
      background: rgba(12, 10, 62, 0.04);
      border-radius: 16px;
      padding: 0.8rem 0.85rem;
    }

    .focus-stat-label {
      color: var(--slate);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .focus-stat-value {
      color: var(--ink);
      font-size: 1.1rem;
      font-weight: 700;
      line-height: 1.1;
      margin-top: 0.35rem;
    }

    .story-list {
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }

    .story-row {
      align-items: flex-start;
      border-top: 1px solid var(--line);
      display: flex;
      gap: 0.9rem;
      justify-content: space-between;
      padding-top: 0.8rem;
    }

    .story-row:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .story-row-copy {
      min-width: 0;
    }

    .story-row-title {
      color: var(--ink);
      font-size: 0.96rem;
      font-weight: 700;
      line-height: 1.35;
    }

    .story-row-meta {
      color: var(--slate);
      font-size: 0.88rem;
      line-height: 1.45;
      margin-top: 0.22rem;
    }

    .delta-pill {
      border-radius: 999px;
      display: inline-flex;
      font-size: 0.78rem;
      font-weight: 700;
      line-height: 1;
      padding: 0.42rem 0.68rem;
      white-space: nowrap;
    }

    .delta-pill.up {
      background: rgba(77, 108, 250, 0.12);
      color: var(--ink-soft);
    }

    .delta-pill.down {
      background: rgba(186, 39, 74, 0.12);
      color: var(--coral);
    }

    .delta-pill.flat {
      background: rgba(94, 104, 135, 0.12);
      color: var(--slate);
    }

    .table-wrap {
      overflow-x: auto;
    }

    .dashboard-table {
      border-collapse: separate;
      border-spacing: 0;
      margin-bottom: 0;
      width: 100%;
    }

    .dashboard-table thead th {
      border-bottom: 1px solid var(--line);
      color: var(--slate);
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      padding: 0.8rem 0.85rem;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .dashboard-table tbody td {
      border-bottom: 1px solid rgba(12, 10, 62, 0.05);
      padding: 0.82rem 0.85rem;
      vertical-align: middle;
    }

    .dashboard-table tbody tr:last-child td {
      border-bottom: 0;
    }

    .dashboard-table tbody tr.table-highlight {
      background: rgba(77, 108, 250, 0.07);
    }

    .rank-col {
      color: var(--slate);
      font-weight: 700;
      white-space: nowrap;
    }

    .county-col {
      min-width: 180px;
    }

    .county-name {
      color: var(--ink);
      font-weight: 700;
      line-height: 1.25;
    }

    .row-chip {
      background: rgba(12, 10, 62, 0.08);
      border-radius: 999px;
      color: var(--ink);
      display: inline-flex;
      font-size: 0.72rem;
      font-weight: 700;
      margin-top: 0.35rem;
      padding: 0.25rem 0.5rem;
    }

    .table-metric-cell {
      min-width: 170px;
    }

    .table-metric-value {
      color: var(--ink);
      font-weight: 700;
      margin-bottom: 0.35rem;
    }

    .table-metric-track {
      background: rgba(12, 10, 62, 0.08);
      border-radius: 999px;
      height: 8px;
      overflow: hidden;
    }

    .table-metric-fill {
      background: linear-gradient(90deg, #4d6cfa 0%, #c78a2c 100%);
      border-radius: inherit;
      height: 100%;
    }

    .delta-col,
    .employment-col {
      white-space: nowrap;
    }

    .bar-list {
      display: flex;
      flex-direction: column;
      gap: 0.9rem;
    }

    .bar-row {
      border-top: 1px solid var(--line);
      padding-top: 0.9rem;
    }

    .bar-row:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .bar-row-top {
      align-items: baseline;
      display: flex;
      gap: 1rem;
      justify-content: space-between;
    }

    .bar-row-label {
      color: var(--ink);
      font-size: 0.95rem;
      font-weight: 700;
      line-height: 1.35;
      max-width: 68%;
    }

    .bar-row-value {
      color: var(--ink);
      font-size: 0.9rem;
      font-weight: 700;
      white-space: nowrap;
    }

    .bar-row-bottom {
      margin-top: 0.42rem;
    }

    .bar-track {
      background: rgba(12, 10, 62, 0.08);
      border-radius: 999px;
      height: 9px;
      overflow: hidden;
    }

    .bar-fill {
      background: linear-gradient(90deg, rgba(77, 108, 250, 0.92) 0%, rgba(186, 39, 74, 0.92) 100%);
      border-radius: inherit;
      height: 100%;
    }

    .bar-row-meta {
      color: var(--slate);
      font-size: 0.86rem;
      line-height: 1.4;
      margin-top: 0.45rem;
    }

    .trend-chart {
      background:
        radial-gradient(circle at top right, rgba(77, 108, 250, 0.08), transparent 34%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(246, 247, 251, 0.92) 100%);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 0.8rem 0.9rem 0.3rem;
    }

    .trend-chart svg {
      display: block;
      height: auto;
      width: 100%;
    }

    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 0.85rem 1.15rem;
      margin-top: 0.8rem;
    }

    .chart-legend-item {
      align-items: center;
      display: inline-flex;
      gap: 0.55rem;
    }

    .chart-legend-line {
      border-top: 3px solid var(--legend-color);
      display: inline-block;
      width: 22px;
    }

    .chart-legend-line.dashed {
      border-top-style: dashed;
    }

    .chart-legend-label {
      color: var(--slate);
      font-size: 0.88rem;
      font-weight: 600;
    }

    .about-grid {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .about-block {
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 1rem 1.05rem;
    }

    .about-block h4 {
      color: var(--ink);
      font-family: "Montserrat", sans-serif;
      font-size: 1rem;
      margin: 0 0 0.4rem;
    }

    .about-block p {
      color: var(--slate);
      font-size: 0.94rem;
      line-height: 1.6;
      margin: 0;
    }

    .empty-state {
      color: var(--slate);
      font-size: 0.94rem;
      line-height: 1.55;
      padding: 0.4rem 0;
    }

    @media (max-width: 1200px) {
      .metrics-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 991px) {
      .hero-highlight-grid,
      .focus-grid,
      .about-grid {
        grid-template-columns: 1fr;
      }

      .map-frame {
        min-height: 560px;
      }
    }

    @media (max-width: 767px) {
      .bslib-sidebar-layout {
        padding: 0.75rem;
      }

      .hero-banner {
        padding: 1.3rem 1.2rem;
      }

      .metrics-grid {
        grid-template-columns: 1fr;
      }

      .bar-row-top,
      .story-row {
        flex-direction: column;
      }

      .bar-row-label {
        max-width: none;
      }
    }
    """,
)

with ui.layout_sidebar(fillable=True):
    with ui.sidebar(
        open="desktop",
        width="330px",
        title="Regional AI Exposure Atlas",
        class_="control-sidebar",
    ):
        ui.tags.div("Narrative dashboard", class_="sidebar-kicker")
        ui.tags.p(
            "Move through time, switch occupation levels, and keep one county in focus while the rest of the story updates around it.",
            class_="sidebar-copy",
        )

        ui.input_slider(
            "year",
            "Year",
            min=FIRST_YEAR,
            max=LATEST_YEAR,
            value=LATEST_YEAR,
            step=1,
            sep="",
        )
        ui.input_select("level", "Occupation level", LEVEL_OPTIONS, selected="SSYK4")
        ui.input_select(
            "metric",
            "Exposure metric",
            METRIC_OPTIONS,
            selected="daioe_genai_wavg",
        )
        ui.input_selectize(
            "focus_county",
            "County spotlight",
            COUNTY_OPTIONS,
            selected=DEFAULT_FOCUS_COUNTY,
        )
        ui.input_selectize(
            "compare_county",
            "Comparison county",
            COUNTY_OPTIONS,
            selected=DEFAULT_COMPARE_COUNTY,
        )

        ui.tags.div(
            ui.tags.div("How to read it", class_="sidebar-note-title"),
            ui.tags.p(
                "County values are employment-weighted averages of the selected DAIOE metric. "
                "Municipality polygons are only used as the geographic scaffold for county colors.",
                class_="sidebar-note-copy",
            ),
            class_="sidebar-note",
        )

        ui.tags.div(
            ui.tags.div("What feels new here", class_="sidebar-note-title"),
            ui.tags.p(
                "The map gives the snapshot, the trends tab shows movement over time, and the county tab reveals which occupations are actually lifting a county's score.",
                class_="sidebar-note-copy",
            ),
            class_="sidebar-note",
        )

    @reactive.calc
    def selected_summary() -> pl.DataFrame:
        frame = county_summary(current_level(), current_year(), current_metric())
        national_value = float(
            frame.select(
                ((pl.col("value") * pl.col("employment")).sum() / pl.col("employment").sum()).alias(
                    "national_value",
                ),
            ).item(),
        )
        return frame.with_row_index("rank", offset=1).with_columns(
            (pl.col("value") - national_value).alias("vs_national"),
        )

    @reactive.calc
    def selected_trend() -> pl.DataFrame:
        return county_timeseries(current_level(), current_metric())

    @reactive.calc
    def selected_national_trend() -> pl.DataFrame:
        return national_timeseries(current_level(), current_metric())

    @reactive.calc
    def selected_rank_shift() -> pl.DataFrame:
        trend = selected_trend()
        baseline = trend.filter(pl.col("year") == FIRST_YEAR).select(
            "county_code",
            pl.col("rank").alias("baseline_rank"),
            pl.col("value").alias("baseline_value"),
        )
        current = trend.filter(pl.col("year") == current_year()).select(
            "county_code",
            "county",
            "employment",
            "rank",
            "value",
        )
        return current.join(baseline, on="county_code").with_columns(
            (pl.col("baseline_rank") - pl.col("rank")).alias("rank_change"),
            (pl.col("value") - pl.col("baseline_value")).alias("value_change"),
        )

    @reactive.calc
    def selected_payload() -> dict:
        frame = selected_summary()
        national_value = float(
            frame.select(
                ((pl.col("value") * pl.col("employment")).sum() / pl.col("employment").sum()).alias(
                    "national_value",
                ),
            ).item(),
        )
        min_value, max_value = county_metric_bounds(current_level(), current_metric())
        county_data = {
            row["county_code"]: {
                "county": row["county"],
                "county_code": row["county_code"],
                "employment": int(row["employment"]),
                "value": float(row["value"]),
            }
            for row in frame.iter_rows(named=True)
        }

        return {
            "data": county_data,
            "focus_code": focus_county_code(),
            "level_label": LEVEL_OPTIONS[current_level()],
            "max": max_value,
            "metric_label": METRIC_OPTIONS[current_metric()],
            "min": min_value,
            "national_value": national_value,
            "year": current_year(),
        }

    @reactive.calc
    def selected_occupation_profile() -> pl.DataFrame:
        return occupation_profile(
            current_level(), current_year(), focus_county_code(), current_metric(),
        )

    @render.ui
    def hero_banner():
        frame = selected_summary()
        trend = selected_trend()
        rank_shift = selected_rank_shift()
        leader = frame.row(0, named=True)
        runner_up = frame.row(1, named=True)
        focus = county_profile_for(
            focus_county_code(),
            summary=frame,
            trend=trend,
            rank_shift=rank_shift,
            year=current_year(),
        )

        return ui.tags.section(
            ui.tags.div("AI-Econ Lab Research Prototype · Beta", class_="hero-eyebrow"),
            ui.tags.h1(
                "A human-friendly view of AI exposure across Swedish counties.",
                class_="hero-title",
            ),
            ui.tags.p(
                "Plainly put: this is a demo built for the Grythyttan retreat (12–13 May 2026). "
                "It uses public SCB employment counts and DAIOE scores mapped from O*NET. "
                "There is no Swedish job-ad NLP or register linkage here yet—that’s the future programme case.",
                class_="hero-copy",
            ),
            ui.tags.div(
                ui.tags.span(str(current_year()), class_="hero-chip"),
                ui.tags.span(LEVEL_OPTIONS[current_level()], class_="hero-chip"),
                ui.tags.span(METRIC_OPTIONS[current_metric()], class_="hero-chip"),
                ui.tags.span(
                    f"Focus: {short_county_name(focus['county'])}",
                    class_="hero-chip",
                ),
                class_="hero-chip-row",
            ),
            ui.tags.div(
                ui.tags.div(
                    ui.tags.div("Current leader", class_="hero-mini-label"),
                    ui.tags.div(short_county_name(leader["county"]), class_="hero-mini-value"),
                    ui.tags.div(
                        f"{format_value(float(leader['value']))} exposure • "
                        f"lead of {format_value(float(leader['value']) - float(runner_up['value']))}",
                        class_="hero-mini-copy",
                    ),
                    class_="hero-mini-card",
                ),
                ui.tags.div(
                    ui.tags.div("Focus county", class_="hero-mini-label"),
                    ui.tags.div(f"Rank #{focus['rank']}", class_="hero-mini-value"),
                    ui.tags.div(
                        f"{format_signed_value(focus['vs_national'])} vs national average",
                        class_="hero-mini-copy",
                    ),
                    class_="hero-mini-card",
                ),
                class_="hero-highlight-grid",
            ),
            class_="hero-banner",
        )

    with ui.navset_card_tab(
        id="tab",
        selected="overview",
        title=ui.tags.div("Regional views", class_="tab-shell-title"),
    ):
        with ui.nav_panel("Overview", value="overview"):
            ui.tags.p(
                "Start with the selected year, then use the focus county to connect the map to a concrete place in the ranking.",
                class_="section-copy",
            )

            @render.ui
            def overview_cards():
                frame = selected_summary()
                trend = selected_trend()
                national = selected_national_trend()
                rank_shift = selected_rank_shift()
                leader = frame.row(0, named=True)
                trailing = frame.tail(1).row(0, named=True)
                runner_up = frame.row(1, named=True)
                focus = county_profile_for(
                    focus_county_code(),
                    summary=frame,
                    trend=trend,
                    rank_shift=rank_shift,
                    year=current_year(),
                )
                national_row = national.filter(pl.col("year") == current_year()).row(0, named=True)
                prior_national = national.filter(pl.col("year") < current_year()).tail(1)
                yoy_text = (
                    "Series start year"
                    if prior_national.height == 0
                    else f"{format_signed_value(float(national_row['value']) - float(prior_national.row(0, named=True)['value']))} versus the previous year"
                )

                return ui.tags.div(
                    make_stat_card(
                        "Leader this year",
                        short_county_name(leader["county"]),
                        f"{format_value(float(leader['value']))} exposure • lead of {format_value(float(leader['value']) - float(runner_up['value']))}",
                        tone="blue",
                    ),
                    make_stat_card(
                        "National benchmark",
                        format_value(float(national_row["value"])),
                        yoy_text,
                        tone="gold",
                    ),
                    make_stat_card(
                        "Focus county",
                        f"#{focus['rank']} {short_county_name(focus['county'])}",
                        f"{format_signed_value(focus['vs_national'])} versus national • {describe_rank_change(focus['rank_change'])}",
                        tone="ink",
                    ),
                    make_stat_card(
                        "Spread across counties",
                        format_value(float(leader["value"]) - float(trailing["value"])),
                        f"{short_county_name(leader['county'])} to {short_county_name(trailing['county'])}",
                        tone="coral",
                    ),
                    class_="metrics-grid",
                )

            with ui.layout_columns(col_widths=(8, 4)):
                with ui.card(class_="app-card"):
                    ui.tags.h3("County choropleth", class_="card-heading")
                    ui.tags.p(
                        "The selected county is outlined so the geographic view and the deeper county profile stay in sync.",
                        class_="card-copy",
                    )

                    @render.ui
                    def county_map():
                        return ui.tags.iframe(
                            srcdoc=build_map_srcdoc(selected_payload()),
                            title="Interactive county exposure map",
                            class_="map-frame",
                        )

                with ui.card(class_="app-card"):
                    ui.tags.h3("Selection snapshot", class_="card-heading")
                    ui.tags.p(
                        "A quick read on the currently selected county and where it sits in the national picture.",
                        class_="card-copy",
                    )

                    @render.ui
                    def overview_focus_panel():
                        frame = selected_summary()
                        trend = selected_trend()
                        rank_shift = selected_rank_shift()
                        focus = county_profile_for(
                            focus_county_code(),
                            summary=frame,
                            trend=trend,
                            rank_shift=rank_shift,
                            year=current_year(),
                        )

                        above_below = "above" if focus["vs_national"] >= 0 else "below"
                        yoy_label = (
                            "No prior year available"
                            if focus["yoy_change"] is None
                            else f"{format_signed_value(focus['yoy_change'])} since the previous year"
                        )

                        return ui.tags.div(
                            ui.tags.p(
                                f"{short_county_name(focus['county'])} sits {format_value(abs(focus['vs_national']))} "
                                f"points {above_below} the national average in {current_year()}. "
                                f"Its current rank is #{focus['rank']}, and its long-run movement is best read as {describe_rank_change(focus['rank_change']).lower()}.",
                                class_="focus-panel-copy",
                            ),
                            ui.tags.div(
                                ui.tags.div(
                                    ui.tags.div("Current score", class_="focus-stat-label"),
                                    ui.tags.div(format_value(focus["value"]), class_="focus-stat-value"),
                                    class_="focus-stat",
                                ),
                                ui.tags.div(
                                    ui.tags.div("Employment", class_="focus-stat-label"),
                                    ui.tags.div(
                                        format_count(focus["employment"]),
                                        class_="focus-stat-value",
                                    ),
                                    class_="focus-stat",
                                ),
                                ui.tags.div(
                                    ui.tags.div("Year-over-year", class_="focus-stat-label"),
                                    ui.tags.div(yoy_label, class_="focus-stat-value"),
                                    class_="focus-stat",
                                ),
                                ui.tags.div(
                                    ui.tags.div("Since baseline", class_="focus-stat-label"),
                                    ui.tags.div(
                                        format_signed_value(focus["value_change"]),
                                        class_="focus-stat-value",
                                    ),
                                    class_="focus-stat",
                                ),
                                class_="focus-grid",
                            ),
                            class_="focus-panel",
                        )

                with ui.card(class_="app-card"):
                    ui.tags.h3("Plain language explainer", class_="card-heading")
                    ui.tags.p(
                        "What you’re seeing: county-level employment-weighted averages of DAIOE exposure scores, using public SCB counts (2014–2024) and O*NET-based DAIOE metrics.",
                        class_="card-copy",
                    )
                    ui.tags.div(
                        ui.tags.div(
                            ui.tags.span("Data sources", class_="story-row-title"),
                            ui.tags.div(
                                "SCB Yrkesregistret employment counts · DAIOE exposure scores (O*NET crosswalk) · No Swedish job-ad NLP or register linkage in this prototype.",
                                class_="story-row-meta",
                            ),
                            class_="story-row-copy",
                        ),
                        ui.tags.div(
                            ui.tags.span("Intended use", class_="story-row-title"),
                            ui.tags.div(
                                "A research prototype for internal demos (Grythyttan retreat) and funding conversations. Not a production monitoring system.",
                                class_="story-row-meta",
                            ),
                            class_="story-row-copy",
                        ),
                        ui.tags.div(
                            ui.tags.span("How to read the colors", class_="story-row-title"),
                            ui.tags.div(
                                "Darker = higher exposure for the selected occupation level. Use the year and level filters first, then pick a county to anchor the story.",
                                class_="story-row-meta",
                            ),
                            class_="story-row-copy",
                        ),
                        class_="story-list",
                    )

                with ui.card(class_="app-card"):
                    ui.tags.h3("Counties climbing the ranking", class_="card-heading")
                    ui.tags.p(
                        "This looks at movement in rank relative to the first year in the series, not just the absolute level.",
                        class_="card-copy",
                    )

                    @render.ui
                    def rank_movers():
                        shifts = selected_rank_shift()
                        if current_year() == FIRST_YEAR:
                            leaders = selected_summary().head(5)
                            rows = []
                            for row in leaders.iter_rows(named=True):
                                rows.append(
                                    ui.tags.div(
                                        ui.tags.div(
                                            ui.tags.div(
                                                short_county_name(row["county"]),
                                                class_="story-row-title",
                                            ),
                                            ui.tags.div(
                                                f"Current rank #{int(row['rank'])} • {format_value(float(row['value']))} exposure",
                                                class_="story-row-meta",
                                            ),
                                            class_="story-row-copy",
                                        ),
                                        make_delta_pill("Baseline view", "flat"),
                                        class_="story-row",
                                    ),
                                )

                            return ui.tags.div(*rows, class_="story-list")

                        movers = shifts.filter(pl.col("rank_change") > 0).sort(
                            ["rank_change", "value"], descending=[True, True],
                        )
                        display = movers.head(5) if movers.height else shifts.sort("rank")
                        rows = []

                        for row in display.iter_rows(named=True):
                            badge = (
                                f"Up {int(row['rank_change'])}"
                                if int(row["rank_change"]) > 0
                                else "No lift"
                            )
                            rows.append(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.div(
                                            short_county_name(row["county"]),
                                            class_="story-row-title",
                                        ),
                                        ui.tags.div(
                                            f"Now rank #{int(row['rank'])} • {format_signed_value(float(row['value_change']))} since {FIRST_YEAR}",
                                            class_="story-row-meta",
                                        ),
                                        class_="story-row-copy",
                                    ),
                                    make_delta_pill(badge, delta_direction(float(row["rank_change"]))),
                                    class_="story-row",
                                ),
                            )

                        return ui.tags.div(*rows, class_="story-list")

            with ui.card(class_="app-card"):
                ui.tags.h3("Current county ranking", class_="card-heading")
                ui.tags.p(
                    "The exposure bar makes the relative distance easier to scan, while the focus county stays highlighted.",
                    class_="card-copy",
                )

                @render.ui
                def county_rank_table():
                    return make_county_table(
                        selected_summary(),
                        highlight_code=focus_county_code(),
                    )

            ui.tags.p(
                "Sources: SCB Yrkesregistret employment (2014–2024). DAIOE scores mapped from O*NET crosswalk. Research prototype — not a production monitoring system.",
                class_="section-copy",
            )

        with ui.nav_panel("Trends", value="trends"):
            ui.tags.p(
                "Use this view to compare the focused county with a second county and see whether the gap is structural or recent.",
                class_="section-copy",
            )

            @render.ui
            def trend_cards():
                frame = selected_summary()
                trend = selected_trend()
                rank_shift = selected_rank_shift()
                national = selected_national_trend()
                focus = county_profile_for(
                    focus_county_code(),
                    summary=frame,
                    trend=trend,
                    rank_shift=rank_shift,
                    year=current_year(),
                )
                national_row = national.filter(pl.col("year") == current_year()).row(0, named=True)

                if compare_county_code() == focus_county_code():
                    compare_card = make_stat_card(
                        "Comparison county",
                        "Pick a second county",
                        "The trends chart currently shows the focus county and the national average only.",
                        tone="coral",
                    )
                    gap_card = make_stat_card(
                        "Focus vs national",
                        format_signed_value(focus["vs_national"]),
                        f"{short_county_name(focus['county'])} against the national benchmark",
                        tone="gold",
                    )
                else:
                    compare = county_profile_for(
                        compare_county_code(),
                        summary=frame,
                        trend=trend,
                        rank_shift=rank_shift,
                        year=current_year(),
                    )
                    gap = focus["value"] - compare["value"]
                    gap_card = make_stat_card(
                        "Current gap",
                        format_signed_value(gap),
                        f"{short_county_name(focus['county'])} minus {short_county_name(compare['county'])}",
                        tone="gold",
                    )
                    compare_card = make_stat_card(
                        "Comparison county",
                        short_county_name(compare["county"]),
                        f"Rank #{compare['rank']} • {format_signed_value(compare['value_change'])} since {FIRST_YEAR}",
                        tone="coral",
                    )

                return ui.tags.div(
                    make_stat_card(
                        "Focus county",
                        short_county_name(focus["county"]),
                        f"Rank #{focus['rank']} • {format_signed_value(focus['value_change'])} since {FIRST_YEAR}",
                        tone="blue",
                    ),
                    compare_card,
                    make_stat_card(
                        "National average",
                        format_value(float(national_row["value"])),
                        f"{LEVEL_OPTIONS[current_level()]} in {current_year()}",
                        tone="ink",
                    ),
                    gap_card,
                    class_="metrics-grid",
                )

            with ui.layout_columns(col_widths=(8, 4)):
                with ui.card(class_="app-card"):
                    ui.tags.h3("Exposure trajectory", class_="card-heading")
                    ui.tags.p(
                        "The national average is shown as the dashed line so county-specific divergence stands out quickly.",
                        class_="card-copy",
                    )

                    @render.ui
                    def trend_chart():
                        trend = selected_trend()
                        national = selected_national_trend()
                        focus_name = COUNTY_OPTIONS[focus_county_code()]
                        compare_name = COUNTY_OPTIONS[compare_county_code()]

                        series = [
                            {
                                "name": focus_name,
                                "color": "#0c0a3e",
                                "points": [
                                    {
                                        "year": int(row["year"]),
                                        "value": float(row["value"]),
                                    }
                                    for row in trend.filter(
                                        pl.col("county_code") == focus_county_code(),
                                    )
                                    .sort("year")
                                    .iter_rows(named=True)
                                ],
                            },
                        ]

                        if compare_county_code() != focus_county_code():
                            series.append(
                                {
                                    "name": compare_name,
                                    "color": "#ba274a",
                                    "points": [
                                        {
                                            "year": int(row["year"]),
                                            "value": float(row["value"]),
                                        }
                                        for row in trend.filter(
                                            pl.col("county_code") == compare_county_code(),
                                        )
                                        .sort("year")
                                        .iter_rows(named=True)
                                    ],
                                },
                            )

                        series.append(
                            {
                                "name": "National average",
                                "color": "#c78a2c",
                                "dash": True,
                                "points": [
                                    {
                                        "year": int(row["year"]),
                                        "value": float(row["value"]),
                                    }
                                    for row in national.iter_rows(named=True)
                                ],
                            },
                        )

                        return ui.tags.div(
                            ui.HTML(build_trend_svg(series)),
                            make_chart_legend(series),
                            class_="trend-chart",
                        )

                with ui.card(class_="app-card"):
                    ui.tags.h3("Trend takeaways", class_="card-heading")
                    ui.tags.p(
                        "A short read of who leads now, who moved more, and whether the focus county is pulling away or converging.",
                        class_="card-copy",
                    )

                    @render.ui
                    def trend_takeaways():
                        frame = selected_summary()
                        trend = selected_trend()
                        rank_shift = selected_rank_shift()
                        focus = county_profile_for(
                            focus_county_code(),
                            summary=frame,
                            trend=trend,
                            rank_shift=rank_shift,
                            year=current_year(),
                        )

                        rows = []
                        rows.append(
                            ui.tags.div(
                                ui.tags.div(
                                    ui.tags.div("Focus county", class_="story-row-title"),
                                    ui.tags.div(
                                        f"{short_county_name(focus['county'])} is currently ranked #{focus['rank']} and sits {format_signed_value(focus['vs_national'])} versus the national average.",
                                        class_="story-row-meta",
                                    ),
                                    class_="story-row-copy",
                                ),
                                make_delta_pill(
                                    describe_rank_change(focus["rank_change"]).replace(
                                        f" since {FIRST_YEAR}", "",
                                    ),
                                    delta_direction(float(focus["rank_change"])),
                                ),
                                class_="story-row",
                            ),
                        )

                        if compare_county_code() != focus_county_code():
                            compare = county_profile_for(
                                compare_county_code(),
                                summary=frame,
                                trend=trend,
                                rank_shift=rank_shift,
                                year=current_year(),
                            )
                            leader = focus if focus["value"] >= compare["value"] else compare
                            lagger = compare if leader is focus else focus
                            rows.append(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.div("Current gap", class_="story-row-title"),
                                        ui.tags.div(
                                            f"{short_county_name(leader['county'])} leads {short_county_name(lagger['county'])} by {format_value(abs(focus['value'] - compare['value']))} points in {current_year()}.",
                                            class_="story-row-meta",
                                        ),
                                        class_="story-row-copy",
                                    ),
                                    make_delta_pill(
                                        short_county_name(leader["county"]),
                                        "up",
                                    ),
                                    class_="story-row",
                                ),
                            )
                            rows.append(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.div("Long-run lift", class_="story-row-title"),
                                        ui.tags.div(
                                            f"{short_county_name(focus['county'])}: {format_signed_value(focus['value_change'])} • {short_county_name(compare['county'])}: {format_signed_value(compare['value_change'])}",
                                            class_="story-row-meta",
                                        ),
                                        class_="story-row-copy",
                                    ),
                                    make_delta_pill(
                                        "Comparative read",
                                        "flat",
                                    ),
                                    class_="story-row",
                                ),
                            )
                        else:
                            rows.append(
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.tags.div("Comparison mode", class_="story-row-title"),
                                        ui.tags.div(
                                            "The same county is selected in both controls, so the chart emphasizes the focus county against the national average.",
                                            class_="story-row-meta",
                                        ),
                                        class_="story-row-copy",
                                    ),
                                    make_delta_pill("Single county", "flat"),
                                    class_="story-row",
                                ),
                            )

                        return ui.tags.div(*rows, class_="story-list")

            with ui.card(class_="app-card"):
                ui.tags.h3("Year-by-year values", class_="card-heading")
                ui.tags.p(
                    "The table makes it easier to read exact values when the lines are close together.",
                    class_="card-copy",
                )

                @render.ui
                def trend_value_table():
                    trend = selected_trend()
                    national = selected_national_trend()
                    focus_label = COUNTY_OPTIONS[focus_county_code()]
                    compare_label = COUNTY_OPTIONS[compare_county_code()]

                    focus_frame = trend.filter(pl.col("county_code") == focus_county_code()).select(
                        "year",
                        pl.col("value").alias("focus_value"),
                    )
                    national_frame = national.select(
                        "year",
                        pl.col("value").alias("national_value"),
                    )

                    joined = focus_frame.join(national_frame, on="year").sort("year", descending=True)
                    headers = [
                        ui.tags.th("Year"),
                        ui.tags.th(focus_label),
                        ui.tags.th("National"),
                        ui.tags.th(f"{focus_label} vs national"),
                    ]

                    if compare_county_code() != focus_county_code():
                        compare_frame = trend.filter(
                            pl.col("county_code") == compare_county_code(),
                        ).select(
                            "year",
                            pl.col("value").alias("compare_value"),
                        )
                        joined = joined.join(compare_frame, on="year")
                        headers.insert(2, ui.tags.th(compare_label))
                        headers.append(ui.tags.th(f"{focus_label} vs {compare_label}"))

                    rows = []
                    for row in joined.iter_rows(named=True):
                        cells = [
                            ui.tags.td(str(int(row["year"]))),
                            ui.tags.td(format_value(float(row["focus_value"]))),
                        ]

                        if compare_county_code() != focus_county_code():
                            focus_gap = float(row["focus_value"]) - float(row["compare_value"])
                            cells.append(ui.tags.td(format_value(float(row["compare_value"]))))
                            cells.append(ui.tags.td(format_value(float(row["national_value"]))))
                            cells.append(
                                ui.tags.td(
                                    make_delta_pill(
                                        format_signed_value(
                                            float(row["focus_value"]) - float(row["national_value"]),
                                        ),
                                        delta_direction(
                                            float(row["focus_value"]) - float(row["national_value"]),
                                        ),
                                    ),
                                ),
                            )
                            cells.append(
                                ui.tags.td(
                                    make_delta_pill(
                                        format_signed_value(focus_gap),
                                        delta_direction(focus_gap),
                                    ),
                                ),
                            )
                        else:
                            cells.append(ui.tags.td(format_value(float(row["national_value"]))))
                            cells.append(
                                ui.tags.td(
                                    make_delta_pill(
                                        format_signed_value(
                                            float(row["focus_value"]) - float(row["national_value"]),
                                        ),
                                        delta_direction(
                                            float(row["focus_value"]) - float(row["national_value"]),
                                        ),
                                    ),
                                ),
                            )

                        rows.append(ui.tags.tr(*cells))

                    return ui.tags.div(
                        ui.tags.table(
                            ui.tags.thead(ui.tags.tr(*headers)),
                            ui.tags.tbody(*rows),
                            class_="dashboard-table",
                        ),
                        class_="table-wrap",
                    )

            ui.tags.p(
                "Context: same SCB employment + DAIOE sources as the map. National line is a benchmark, not a target. Gaps help spark the ‘what should we investigate next?’ question.",
                class_="section-copy",
            )

        with ui.nav_panel("County Deep Dive", value="county"):
            ui.tags.p(
                "This view shows what is inside the county score, so you can tell whether exposure comes from a few standout occupations or a broad-based mix.",
                class_="section-copy",
            )

            @render.ui
            def county_detail_cards():
                frame = selected_summary()
                trend = selected_trend()
                rank_shift = selected_rank_shift()
                profile = selected_occupation_profile().sort("score_contribution", descending=True)
                focus = county_profile_for(
                    focus_county_code(),
                    summary=frame,
                    trend=trend,
                    rank_shift=rank_shift,
                    year=current_year(),
                )

                if profile.height == 0:
                    top_driver_title = "No occupation detail"
                    top_driver_detail = "This county has no non-null occupation exposures for the selected metric."
                else:
                    driver = profile.row(0, named=True)
                    top_driver_title = driver["occupation"]
                    top_driver_detail = (
                        f"{format_percent(float(driver['employment_share']))} of county employment • "
                        f"{format_value(float(driver['score_contribution']))} score points"
                    )

                return ui.tags.div(
                    make_stat_card(
                        "County spotlight",
                        short_county_name(focus["county"]),
                        f"Rank #{focus['rank']} • {format_value(focus['value'])} exposure",
                        tone="blue",
                    ),
                    make_stat_card(
                        "Employment base",
                        format_count(focus["employment"]),
                        f"{format_signed_value(focus['vs_national'])} versus the national average",
                        tone="ink",
                    ),
                    make_stat_card(
                        "Top score driver",
                        top_driver_title,
                        top_driver_detail,
                        tone="coral",
                    ),
                    make_stat_card(
                        "Long-run movement",
                        format_signed_value(focus["value_change"]),
                        describe_rank_change(focus["rank_change"]),
                        tone="gold",
                    ),
                    class_="metrics-grid",
                )

            with ui.layout_columns(col_widths=(6, 6)):
                with ui.card(class_="app-card"):
                    ui.tags.h3("Occupations driving the county score", class_="card-heading")
                    ui.tags.p(
                        "These are the occupations contributing the most score points once employment size is taken into account.",
                        class_="card-copy",
                    )

                    @render.ui
                    def driver_bar_list():
                        profile = selected_occupation_profile().sort(
                            "score_contribution", descending=True,
                        )
                        return make_bar_list(
                            profile,
                            label_col="occupation",
                            bar_col="score_contribution",
                            value_builder=lambda row: f"{format_value(float(row['score_contribution']))} points",
                            meta_builder=lambda row: (
                                f"{format_percent(float(row['employment_share']))} of county employment • "
                                f"{METRIC_OPTIONS[current_metric()]} {format_value(float(row['value']))}"
                            ),
                        )

                with ui.card(class_="app-card"):
                    ui.tags.h3("Highest exposure occupations", class_="card-heading")
                    ui.tags.p(
                        "This version filters to occupations with at least 100 workers so the ranking does not get dominated by tiny categories.",
                        class_="card-copy",
                    )

                    @render.ui
                    def top_exposure_bar_list():
                        profile = selected_occupation_profile()
                        filtered = profile.filter(pl.col("employment") >= 100)
                        if filtered.height == 0:
                            filtered = profile

                        filtered = filtered.sort("value", descending=True)

                        return make_bar_list(
                            filtered,
                            label_col="occupation",
                            bar_col="value",
                            value_builder=lambda row: f"{format_value(float(row['value']))} exposure",
                            meta_builder=lambda row: (
                                f"{format_count(int(row['employment']))} workers • "
                                f"{format_percent(float(row['employment_share']))} of county employment"
                            ),
                        )

            with ui.card(class_="app-card"):
                ui.tags.h3("Detailed occupation table", class_="card-heading")
                ui.tags.p(
                    "Sorted by contribution to the county score so the most influential occupations rise to the top.",
                    class_="card-copy",
                )

                @render.ui
                def occupation_detail_table():
                    profile = selected_occupation_profile().sort(
                        "score_contribution", descending=True,
                    )

                    if profile.height == 0:
                        return ui.tags.div(
                            "No occupation detail is available for this county and metric.",
                            class_="empty-state",
                        )

                    rows = []
                    for row in profile.head(12).iter_rows(named=True):
                        rows.append(
                            ui.tags.tr(
                                ui.tags.td(row["occupation"]),
                                ui.tags.td(format_value(float(row["value"]))),
                                ui.tags.td(format_count(int(row["employment"]))),
                                ui.tags.td(format_percent(float(row["employment_share"]))),
                                ui.tags.td(format_value(float(row["score_contribution"]))),
                            ),
                        )

                    return ui.tags.div(
                        ui.tags.table(
                            ui.tags.thead(
                                ui.tags.tr(
                                    ui.tags.th("Occupation"),
                                    ui.tags.th("Exposure"),
                                    ui.tags.th("Employment"),
                                    ui.tags.th("Employment share"),
                                    ui.tags.th("Score contribution"),
                                ),
                            ),
                            ui.tags.tbody(*rows),
                            class_="dashboard-table",
                        ),
                        class_="table-wrap",
                    )

            ui.tags.p(
                "Human note: these breakdowns are descriptive, not prescriptive. They show which occupations lift a county’s score with the data we currently have; they do not predict outcomes.",
                class_="section-copy",
            )

        with ui.nav_panel("About", value="about"):
            ui.tags.p(
                "A little context for what the app is summarizing and why the redesign is structured the way it is.",
                class_="section-copy",
            )

            with ui.card(class_="app-card"):
                ui.tags.h3("What this app is doing now", class_="card-heading")
                ui.tags.p(
                    "Instead of only showing a basic county map, the app now has a clearer analytical flow: snapshot, trend, and explanation — and it carries the “AI-Econ Lab Research Prototype • Beta” label per the memo.",
                    class_="card-copy",
                )

                ui.tags.div(
                    ui.tags.div(
                        ui.tags.h4("Map first", class_="card-heading"),
                        ui.tags.p(
                            "The choropleth gives a fast regional read, with a selected county highlighted so the user always has a concrete anchor.",
                        ),
                        class_="about-block",
                    ),
                    ui.tags.div(
                        ui.tags.h4("Trend second", class_="card-heading"),
                        ui.tags.p(
                            "The comparison tab shows whether a county's position is persistent or changing, using the national average as a stable benchmark line.",
                        ),
                        class_="about-block",
                    ),
                    ui.tags.div(
                        ui.tags.h4("County explanation", class_="card-heading"),
                        ui.tags.p(
                            "The deep dive breaks open the county score and highlights which occupations contribute most once employment shares are considered.",
                        ),
                        class_="about-block",
                    ),
                    ui.tags.div(
                        ui.tags.h4("Method notes", class_="card-heading"),
                        ui.tags.p(
                            f"The parquet file covers all 21 Swedish counties across {FIRST_YEAR} through {LATEST_YEAR}. "
                            "County scores are employment-weighted averages of the selected DAIOE metric, and municipality geometry is used only as a visual scaffold for county-level data. "
                            "No Swedish job-ad NLP, register linkage, or scenario tools are included — those stay on the future programme list.",
                        ),
                        class_="about-block",
                    ),
                    class_="about-grid",
                )
