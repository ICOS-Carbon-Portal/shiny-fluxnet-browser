from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

MAX_SERIES = 6
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#e377c2"]

COLOR_PALETTE = {
    "#1f77b4": "Blue",
    "#d62728": "Red",
    "#2ca02c": "Green",
    "#9467bd": "Purple",
    "#ff7f0e": "Orange",
    "#e377c2": "Pink",
    "#17becf": "Cyan",
    "#8c564b": "Brown",
    "#7f7f7f": "Grey",
    "#bcbd22": "Olive",
    "#000000": "Black",
    "#00008b": "Dark Blue",
    "#006400": "Dark Green",
    "#8b0000": "Dark Red",
    "#ff1493": "Deep Pink",
    "#ffd700": "Gold",
    "#00ced1": "Dark Turquoise",
    "#ff6347": "Tomato",
}

AGG_CHOICES = {
    "raw": "Raw (native resolution)",
    "daily": "Daily average",
    "weekly": "Weekly average",
    "monthly": "Monthly average",
}

LINE_CHOICES = {
    "solid": "Solid",
    "dash": "Dash",
    "dot": "Dot",
    "dashdot": "Dash-dot",
}

CHART_CHOICES = {
    "line": "Line",
    "bar": "Bar (monthly only)",
}

VIEW_MODES = {
    "timeseries": "Time series",
    "tod_30": "Avg by time of day (30 min)",
    "tod_60": "Avg by time of day (hourly)",
    "week": "Avg by week of year",
    "month": "Avg by month of year",
}

# ---------------------------------------------------------------------------
# ICOS Carbon Portal helpers
# ---------------------------------------------------------------------------

ICOS_SPARQL_ENDPOINT = "https://meta.icos-cp.eu/sparql"

ICOS_QUERY = """\
prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
prefix prov: <http://www.w3.org/ns/prov#>
prefix xsd: <http://www.w3.org/2001/XMLSchema#>
prefix geo: <http://www.opengis.net/ont/geosparql#>
select ?dobj ?fileName
from <http://meta.icos-cp.eu/resources/cpmeta/>
from <http://meta.icos-cp.eu/resources/icos/>
from <http://meta.icos-cp.eu/resources/extrastations/>
where {
    VALUES ?spec {<http://meta.icos-cp.eu/resources/cpmeta/etcL2Fluxnet>}
    ?dobj cpmeta:hasObjectSpec ?spec .
    BIND(EXISTS{[] cpmeta:isNextVersionOf ?dobj} AS ?hasNextVersion)
    ?dobj cpmeta:hasSizeInBytes ?size .
    ?dobj cpmeta:hasName ?fileName .
    ?dobj cpmeta:wasSubmittedBy/prov:endedAtTime ?submTime .
    ?dobj cpmeta:hasStartTime | (cpmeta:wasAcquiredBy / prov:startedAtTime) ?timeStart .
    ?dobj cpmeta:hasEndTime | (cpmeta:wasAcquiredBy / prov:endedAtTime) ?timeEnd .
    FILTER NOT EXISTS {[] cpmeta:isNextVersionOf ?dobj}
}
order by desc(?submTime)
"""


def icos_query_files() -> dict[str, str]:
    """Run the SPARQL query and return {dobj_url: fileName} dict."""
    resp = requests.post(
        ICOS_SPARQL_ENDPOINT,
        data={"query": ICOS_QUERY},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]["bindings"]
    return {r["dobj"]["value"]: r["fileName"]["value"] for r in results}


def _read_csv_flexible(buf: io.BytesIO) -> pd.DataFrame:
    """Try comma then semicolon separator; raise on total failure."""
    try:
        df = pd.read_csv(buf)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    buf.seek(0)
    return pd.read_csv(buf, sep=";")


def icos_download_csv(dobj_url: str) -> pd.DataFrame:
    """Download a data object from the ICOS Carbon Portal and return a DataFrame.

    Uses the ``licence_accept`` endpoint which auto-accepts the data license.
    The *dobj_url* looks like ``https://meta.icos-cp.eu/objects/<hash_id>``.
    The download URL is ``https://data.icos-cp.eu/licence_accept?ids=["<hash_id>"]``.
    ETC L2 Fluxnet data arrives as ZIP archives containing one CSV file.
    """
    # Extract the hash id from the end of the dobj URL
    hash_id = dobj_url.rstrip("/").rsplit("/", 1)[-1]
    download_url = f'https://data.icos-cp.eu/licence_accept?ids=%5B%22{hash_id}%22%5D'
    resp = requests.get(
        download_url,
        timeout=120,
    )
    resp.raise_for_status()

    raw = io.BytesIO(resp.content)

    # If the payload is a zip, extract the first CSV inside it.
    if zipfile.is_zipfile(raw):
        raw.seek(0)
        with zipfile.ZipFile(raw) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            target = csv_names[0] if csv_names else zf.namelist()[0]
            with zf.open(target) as f:
                return _read_csv_flexible(io.BytesIO(f.read()))

    # Not a zip – try reading as CSV directly.
    raw.seek(0)
    return _read_csv_flexible(raw)


def _looks_like_datetime(val) -> bool:
    """Return True if *val* can plausibly be parsed as a timestamp."""
    if pd.isna(val):
        return False
    s = str(val).strip()
    if not s:
        return False
    # YYYYMMDDHHMM (12 digits) or YYYYMMDD (8 digits)
    if re.fullmatch(r"\d{8}(\d{4})?", s):
        return True
    try:
        pd.to_datetime(s)
        return True
    except Exception:
        return False


def guess_datetime_col(df: pd.DataFrame) -> str:
    """Guess which column holds timestamps by inspecting the first row.

    Preference: first column, then any column whose first value parses as a datetime.
    """
    cols = list(df.columns)
    if not cols:
        return cols[0] if cols else ""
    # Check first column first (most common layout)
    first_val = df.iloc[0][cols[0]]
    if _looks_like_datetime(first_val):
        return cols[0]
    # Scan remaining columns
    for c in cols[1:]:
        if _looks_like_datetime(df.iloc[0][c]):
            return c
    # Fallback: first column
    return cols[0]


def smart_parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a Series to datetime, handling YYYYMMDDHHMM / YYYYMMDD numeric formats."""
    s = series.astype(str).str.strip()
    # Detect compact numeric timestamps: 12-digit YYYYMMDDHHMM or 8-digit YYYYMMDD
    is_compact = s.str.fullmatch(r"\d{8}(\d{4})?")
    if is_compact.any():
        def _parse_compact(v: str):
            v = v.strip()
            if len(v) == 12:  # YYYYMMDDHHMM
                return pd.to_datetime(v, format="%Y%m%d%H%M", errors="coerce")
            if len(v) == 8:   # YYYYMMDD
                return pd.to_datetime(v, format="%Y%m%d", errors="coerce")
            return pd.to_datetime(v, errors="coerce")
        return s.apply(_parse_compact)
    # Fall back to pandas general parser (handles ISO and many other layouts)
    return pd.to_datetime(series, errors="coerce")


def series_controls(slot: int) -> ui.Tag:
    return ui.card(
        ui.card_header(f"Series {slot}"),
        ui.input_select(f"col_{slot}", "Column", choices={"": "(none)"}, selected=""),
        ui.input_select(f"agg_{slot}", "Averaging", choices=AGG_CHOICES, selected="raw"),
        ui.input_select(f"chart_{slot}", "Chart type", choices=CHART_CHOICES, selected="line"),
        ui.input_select(f"dash_{slot}", "Line type", choices=LINE_CHOICES, selected="solid"),
        ui.input_select(f"color_{slot}", "Color", choices=COLOR_PALETTE, selected=DEFAULT_COLORS[slot - 1]),
        ui.input_select(
            f"yaxis_{slot}",
            "Y-axis",
            choices={"1": "Axis 1", "2": "Axis 2", "3": "Axis 3", "4": "Axis 4"},
            selected="1",
        ),
    )


_COLOR_OPTION_CSS = "\n".join(
    'select[id^="color_"] option[value="{hex}"] {{ background-color: {hex}; color: {fg}; }}'.format(
        hex=hex_val,
        fg="#fff" if hex_val in ("#000000", "#00008b", "#006400", "#8b0000", "#8c564b") else "#000",
    )
    for hex_val in COLOR_PALETTE
)


app_ui = ui.page_fluid(
    ui.div(
        ui.tags.img(src="icos_logo.png", height="40px", style="vertical-align: middle; margin-right: 10px;"),
        ui.tags.span("FLUXNET data browser", style="font-size: 14pt !important; font-weight: bold; vertical-align: middle;"),
        style="display: flex; align-items: center; margin-bottom: 0.2rem;",
    ),
    ui.tags.style(
        "body, .form-control, .form-select, label, .card-header,"
        " .shiny-input-container, .selectize-input, .selectize-dropdown,"
        " .btn, .form-check-label, h5 {"
        "   font-size: 10pt !important;"
        " }"
        " h2 { font-size: 14pt !important; margin-bottom: 0.2rem !important; }"
        " .card { margin-bottom: 0.15rem !important; }"
        " .card-header { padding: 0.1rem 0.3rem !important; }"
        " .card-body { padding: 0.1rem 0.3rem !important; }"
        " .card-body > * { margin-bottom: 0 !important; margin-top: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; }"
        " .card-body .shiny-input-container { margin-bottom: 0 !important; margin-top: 1px !important; }"
        " .card-body .form-group { margin-bottom: 0 !important; }"
        " .card-body select, .card-body input { margin-bottom: 0 !important; margin-top: 0 !important; }"
        " .form-group, .shiny-input-container { margin-bottom: 0 !important; }"
        " .form-control, .form-select { padding: 0.05rem 0.2rem !important; height: auto !important; min-height: 0 !important; line-height: 1.2 !important; }"
        " label { margin-bottom: 0 !important; line-height: 1.2 !important; }"
        " .row { --bs-gutter-x: 0.4rem !important; --bs-gutter-y: 0.2rem !important; }"
        " .col, [class^='col-'] { padding-left: 0.2rem !important; padding-right: 0.2rem !important; }"
        " hr { margin: 0.2rem 0 !important; }"
        " .sidebar { font-size: 10pt !important; padding: 0.3rem !important; }"
        " .sidebar .form-group, .sidebar .shiny-input-container { margin-bottom: 0 !important; margin-top: 1px !important; }"
        " .sidebar > * { margin-bottom: 0 !important; margin-top: 1px !important; }"
        " .sidebar label { margin-bottom: 0 !important; }"
        " .sidebar select, .sidebar input { margin-bottom: 0 !important; margin-top: 0 !important; }"
        " .sidebar .form-check { margin-bottom: 0 !important; margin-top: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; min-height: 0 !important; line-height: 1 !important; }"
        " .sidebar .form-group.shiny-input-container { margin-bottom: 0 !important; margin-top: 0 !important; padding: 0 !important; }"
        " .sidebar .row { margin-top: 0 !important; margin-bottom: 0 !important; --bs-gutter-y: 0 !important; }"
        " .sidebar .row .col-sm-6 .shiny-input-container label { display: none !important; }"
        " .sidebar .row .col-sm-6 .shiny-input-container { margin-top: 0 !important; margin-bottom: 0 !important; }"
        " .sidebar .row .col-sm-6 .form-control { margin-top: 0 !important; padding-top: 1px !important; padding-bottom: 1px !important; }"
        " .yax-group { margin-bottom: 2px !important; margin-top: 0 !important; }"
        " .yax-group > * { margin-bottom: 0 !important; margin-top: 0 !important; }"
        " .yax-group .form-check { margin-bottom: -2px !important; }"
        " .yax-group .row { margin-top: -2px !important; }"
        " .sidebar h5 { margin-bottom: 0 !important; margin-top: 2px !important; font-size: 10pt !important; font-weight: bold !important; }"
        " .sidebar hr { margin: 2px 0 !important; }"
        " #ts_plot { min-height: 50vh !important; height: 50vh !important; }"
        " #ts_plot > div, #ts_plot iframe,"
        " #ts_plot .plotly, #ts_plot .plot-container,"
        " #ts_plot .html-widget {"
        "   width: 100% !important; height: 100% !important; min-height: 50vh !important;"
        " }"
        " .sidebar { background-color: #F8F8F8 !important; }"
        " .card { background-color: #F8F8F8 !important; }"
        " .card-header { background-color: #00ABC9 !important; color: #fff !important; }"
        " .btn { background-color: #00ABC9 !important; color: #fff !important; border-color: #00ABC9 !important; }"
        " .btn:hover { background-color: #008fa8 !important; border-color: #008fa8 !important; }"
        + _COLOR_OPTION_CSS
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("ICOS Carbon Portal"),
            ui.input_action_button("icos_query", "Query ICOS files", class_="btn-sm btn-outline-primary"),
            ui.output_text_verbatim("icos_status"),
            ui.input_select("icos_file", "ICOS file", choices={}),
            ui.input_action_button("icos_load", "Download & load", class_="btn-sm btn-outline-success"),
            ui.hr(),
            ui.h5("Or upload local file"),
            ui.input_file("data_file", "Data file (.csv, .xlsx, .zip)", accept=[".csv", ".xlsx", ".xls", ".zip"]),
            ui.input_select("dt_col", "Datetime column", choices={}),
            ui.hr(),
            ui.h5("View mode"),
            ui.input_select("view_mode", "Plot type", choices=VIEW_MODES, selected="timeseries"),
            ui.hr(),
            ui.h5("Time interval filter"),
            ui.input_text("start_ts", "Start (optional)", placeholder="YYYY-MM-DD HH:MM"),
            ui.input_text("end_ts", "End (optional)", placeholder="YYYY-MM-DD HH:MM"),
            ui.hr(),
            ui.h5("X-axis scale"),
            ui.input_checkbox("x_auto", "Auto scale x-axis", value=True),
            ui.input_text("x_min", "Manual x min", placeholder="YYYY-MM-DD HH:MM"),
            ui.input_text("x_max", "Manual x max", placeholder="YYYY-MM-DD HH:MM"),
            ui.hr(),
            ui.h5("Y-axis scales"),
            *[
                ui.div(
                    ui.input_checkbox(f"y{ax}_auto", f"Axis {ax} auto", value=True),
                    ui.row(
                        ui.column(6, ui.input_text(f"y{ax}_min", None, placeholder="Min")),
                        ui.column(6, ui.input_text(f"y{ax}_max", None, placeholder="Max")),
                    ),
                    class_="yax-group",
                )
                for ax in range(1, 5)
            ],
            width=360,
        ),
        ui.div(
            ui.output_text_verbatim("status"),
            ui.row(
                ui.column(10),
                ui.column(2, ui.download_button("download_pdf", "Export PDF", class_="btn-sm btn-outline-secondary")),
            ),
            output_widget("ts_plot", height="50vh"),
            ui.hr(),
            ui.row(*[ui.column(4, series_controls(i)) for i in range(1, MAX_SERIES + 1)]),
        ),
    ),
)


def _read_from_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    """Read data from a zip archive.

    pd.read_csv handles zip compression natively.  If the archive only
    contains Excel files, fall back to extracting the first one.
    """
    try:
        return pd.read_csv(zip_path, compression="zip")
    except Exception:
        pass
    # Fallback: look for an Excel file inside the archive
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith((".xlsx", ".xls")):
                with zf.open(name) as f:
                    return pd.read_excel(io.BytesIO(f.read()))
    return None


def server(input, output, session):
    # --- ICOS state ---
    _icos_files: reactive.Value[dict[str, str]] = reactive.value({})
    _icos_df: reactive.Value[Optional[pd.DataFrame]] = reactive.value(None)
    _icos_name: reactive.Value[str] = reactive.value("")
    # Track temp file paths so we can delete previous downloads/uploads
    _prev_temp_paths: reactive.Value[list[str]] = reactive.value([])
    # Store current figure for PDF export
    _current_fig: reactive.Value[Optional[go.Figure]] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.icos_query)
    def _do_icos_query():
        try:
            files = icos_query_files()
            _icos_files.set(files)
            choices = {url: name for url, name in files.items()}
            ui.update_select("icos_file", choices=choices)
        except Exception as exc:
            _icos_files.set({})
            ui.notification_show(f"ICOS query failed: {exc}", type="error")

    @output
    @render.text
    def icos_status() -> str:
        n = len(_icos_files.get())
        if n == 0:
            return "Click 'Query ICOS files' to search."
        return f"{n} files found."

    def _cleanup_temp_files():
        """Delete previously tracked temp files from disk."""
        for p in _prev_temp_paths.get():
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        _prev_temp_paths.set([])

    @reactive.effect
    @reactive.event(input.icos_load)
    def _do_icos_download():
        url = input.icos_file()
        if not url:
            return
        try:
            _cleanup_temp_files()
            df = icos_download_csv(url)
            _icos_df.set(df)
            files = _icos_files.get()
            _icos_name.set(files.get(url, "ICOS"))
            ui.notification_show(f"Loaded {len(df)} rows from ICOS.", type="message")
        except Exception as exc:
            ui.notification_show(f"Download failed: {exc}", type="error")

    @reactive.calc
    def raw_df() -> Optional[pd.DataFrame]:
        # ICOS data takes priority if loaded
        icos = _icos_df.get()
        if icos is not None and not icos.empty:
            return icos

        file_info = input.data_file()
        if not file_info:
            return None

        # Track the upload temp path so it can be cleaned up later
        upload_path = file_info[0]["datapath"]
        prev = _prev_temp_paths.get()
        if upload_path not in prev:
            _prev_temp_paths.set(prev + [upload_path])

        file_path = Path(upload_path)
        # Use the *original* filename for extension detection because
        # Shiny copies uploads to a temp path that may lose the extension.
        original_name = file_info[0].get("name", "")
        suffix = Path(original_name).suffix.lower() if original_name else file_path.suffix.lower()

        if suffix == ".zip":
            df = _read_from_zip(file_path)
        elif suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        else:
            return None

        return df

    @reactive.effect
    def _update_column_inputs() -> None:
        df = raw_df()
        if df is None or df.empty:
            return

        all_cols = [str(c) for c in df.columns]
        numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
        if not numeric_cols:
            numeric_cols = all_cols

        best_dt = guess_datetime_col(df)
        ui.update_select("dt_col", choices={c: c for c in all_cols}, selected=best_dt)

        col_choices = {"": "(none)"}
        col_choices.update({c: c for c in numeric_cols})

        for slot in range(1, MAX_SERIES + 1):
            with reactive.isolate():
                current = input[f"col_{slot}"]()
            # Keep current selection if the column still exists in the new file
            sel = current if current and current in col_choices else ""
            ui.update_select(f"col_{slot}", choices=col_choices, selected=sel)

    @reactive.effect
    @reactive.event(input.view_mode)
    def _update_agg_choices() -> None:
        """Enable/disable averaging based on view mode (only reacts to view_mode)."""
        view = input.view_mode()
        is_profile = view != "timeseries"
        for slot in range(1, MAX_SERIES + 1):
            if is_profile:
                ui.update_select(
                    f"agg_{slot}",
                    choices={"raw": "(not applicable)"},
                    selected="raw",
                )
                ui.update_select(f"chart_{slot}", choices={"line": "Line", "bar": "Bar"}, selected="line")
            else:
                with reactive.isolate():
                    agg = input[f"agg_{slot}"]()
                ui.update_select(
                    f"agg_{slot}",
                    choices=AGG_CHOICES,
                    selected=agg if agg in AGG_CHOICES else "raw",
                )
                ui.update_select(f"chart_{slot}", choices={"line": "Line"}, selected="line")

    @reactive.effect
    def _update_chart_for_monthly() -> None:
        """Allow Bar when agg is monthly (only in timeseries mode)."""
        view = input.view_mode()
        if view != "timeseries":
            return  # handled by _update_agg_choices
        for slot in range(1, MAX_SERIES + 1):
            agg = input[f"agg_{slot}"]()
            if agg == "monthly":
                choices = {"line": "Line", "bar": "Bar"}
            else:
                choices = {"line": "Line"}
            with reactive.isolate():
                current = input[f"chart_{slot}"]()
            sel = current if current in choices else "line"
            ui.update_select(f"chart_{slot}", choices=choices, selected=sel)

    @reactive.calc
    def parsed_df() -> Optional[pd.DataFrame]:
        df = raw_df()
        if df is None or df.empty:
            return None

        dt_col = input.dt_col()
        if not dt_col or dt_col not in df.columns:
            return None

        out = df.copy()
        out[dt_col] = smart_parse_datetime(out[dt_col])
        out = out.dropna(subset=[dt_col]).sort_values(dt_col)

        return out

    def parse_opt_datetime(value: str):
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        return pd.to_datetime(value, errors="coerce")

    @reactive.calc
    def filtered_df() -> Optional[pd.DataFrame]:
        df = parsed_df()
        if df is None or df.empty:
            return None

        dt_col = input.dt_col()
        start = parse_opt_datetime(input.start_ts())
        end = parse_opt_datetime(input.end_ts())

        if start is not None and not pd.isna(start):
            df = df[df[dt_col] >= start]
        if end is not None and not pd.isna(end):
            df = df[df[dt_col] <= end]

        return df

    @output
    @render.text
    def status() -> str:
        df = filtered_df()
        if df is None:
            return "Upload a CSV/XLSX file to begin."
        if df.empty:
            return "No rows available after datetime parsing/filtering."

        dt_col = input.dt_col()
        min_ts = df[dt_col].min()
        max_ts = df[dt_col].max()
        return f"Rows: {len(df):,} | Range: {min_ts} -> {max_ts}"

    def aggregate_series(frame: pd.DataFrame, dt_col: str, value_col: str, agg: str) -> pd.DataFrame:
        series = frame[[dt_col, value_col]].copy()
        # Drop rows where datetime is missing, but keep NaN values so gaps show in plots
        series = series.dropna(subset=[dt_col])
        if series.empty:
            return series

        if agg == "raw":
            return series

        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
        }

        freq = freq_map.get(agg)
        if freq is None:
            return series

        grouped = (
            series.set_index(dt_col)[value_col]
            .resample(freq)
            .mean()
            .dropna()
            .rename(value_col)
            .reset_index()
        )
        return grouped

    def profile_series(
        frame: pd.DataFrame, dt_col: str, value_col: str, view: str
    ) -> tuple[pd.Series, pd.Series, str]:
        """Aggregate *value_col* into a profile (time-of-day / week / month).

        Returns (x_values, y_values, x_label).
        """
        data = frame[[dt_col, value_col]].dropna().copy()
        ts = data[dt_col]

        if view == "tod_30":
            # Half-hour bin: 0, 0.5, 1, …, 23.5
            bucket = ts.dt.hour + (ts.dt.minute // 30) * 0.5
            grouped = data.groupby(bucket)[value_col].mean()
            labels = [f"{int(h)}:{int((h % 1)*60):02d}" for h in grouped.index]
            return pd.Series(labels, index=grouped.index), grouped, "Time of day"

        if view == "tod_60":
            bucket = ts.dt.hour
            grouped = data.groupby(bucket)[value_col].mean()
            labels = [f"{int(h)}:00" for h in grouped.index]
            return pd.Series(labels, index=grouped.index), grouped, "Time of day"

        if view == "week":
            bucket = ts.dt.isocalendar().week.astype(int)
            grouped = data.groupby(bucket)[value_col].mean().sort_index()
            return grouped.index.to_series(), grouped, "Week of year"

        if view == "month":
            bucket = ts.dt.month
            grouped = data.groupby(bucket)[value_col].mean().sort_index()
            month_names = [
                "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]
            labels = [month_names[int(m)] for m in grouped.index]
            return pd.Series(labels, index=grouped.index), grouped, "Month"

        return pd.Series(dtype=object), pd.Series(dtype=float), ""

    @reactive.calc
    def _plot_inputs():
        """Collect all inputs that affect the plot into a single dict."""
        series = []
        for slot in range(1, MAX_SERIES + 1):
            series.append({
                "col": input[f"col_{slot}"](),
                "agg": input[f"agg_{slot}"](),
                "chart": input[f"chart_{slot}"](),
                "dash": input[f"dash_{slot}"](),
                "color": input[f"color_{slot}"](),
                "yaxis": input[f"yaxis_{slot}"](),
            })
        yaxes = {}
        for ax in range(1, 5):
            yaxes[ax] = {
                "auto": input[f"y{ax}_auto"](),
                "min": input[f"y{ax}_min"](),
                "max": input[f"y{ax}_max"](),
            }
        return {
            "view_mode": input.view_mode(),
            "x_auto": input.x_auto(),
            "x_min": input.x_min(),
            "x_max": input.x_max(),
            "yaxes": yaxes,
            "series": series,
        }

    # Manual debounce: copy _plot_inputs into a reactive.Value after 1.5s of inactivity
    @reactive.effect
    def _update_yaxis_ranges():
        """Populate y-axis min/max fields with computed data ranges when auto is on."""
        df = filtered_df()
        if df is None or df.empty:
            return
        dt_col = input.dt_col()
        view_mode = input.view_mode()
        is_profile = view_mode != "timeseries"

        # Collect min/max per axis from plotted series
        axis_mins: dict[int, float] = {}
        axis_maxs: dict[int, float] = {}
        for slot in range(1, MAX_SERIES + 1):
            col = input[f"col_{slot}"]()
            if not col or col not in df.columns:
                continue
            agg = input[f"agg_{slot}"]()
            try:
                ax = int(input[f"yaxis_{slot}"]())
            except Exception:
                ax = 1
            ax = max(1, min(4, ax))

            # QC filtering
            qc_col = f"{col}_QC"
            sdf = df[~df[qc_col].isin([2, 3])] if qc_col in df.columns else df

            if is_profile:
                _, y_vals, _ = profile_series(sdf, dt_col, col, view_mode)
                vals = y_vals.dropna()
            else:
                agg_data = aggregate_series(sdf, dt_col, col, agg)
                if agg_data.empty:
                    continue
                vals = agg_data[col].dropna()

            if vals.empty:
                continue
            vmin = float(vals.min())
            vmax = float(vals.max())
            axis_mins[ax] = min(axis_mins.get(ax, vmin), vmin)
            axis_maxs[ax] = max(axis_maxs.get(ax, vmax), vmax)

        for ax in range(1, 5):
            with reactive.isolate():
                is_auto = input[f"y{ax}_auto"]()
            if is_auto and ax in axis_mins:
                # Round to reasonable precision
                lo = axis_mins[ax]
                hi = axis_maxs[ax]
                # Add a small 5% padding like Plotly does
                span = hi - lo if hi != lo else abs(hi) * 0.1 or 1.0
                lo_padded = lo - span * 0.05
                hi_padded = hi + span * 0.05
                ui.update_text(f"y{ax}_min", value=f"{lo_padded:.4g}")
                ui.update_text(f"y{ax}_max", value=f"{hi_padded:.4g}")
            elif is_auto:
                ui.update_text(f"y{ax}_min", value="")
                ui.update_text(f"y{ax}_max", value="")

    @output
    @render_widget
    def ts_plot():
        params = _plot_inputs()
        df = filtered_df()
        fig = go.Figure()

        if df is None or df.empty:
            fig.update_layout(
                template="plotly_white",
                title="Time Series",
                xaxis_title="Datetime",
                yaxis_title="Value",
                autosize=True,
            )
            return fig

        dt_col = input.dt_col()
        view_mode = params["view_mode"]
        is_profile = view_mode != "timeseries"

        # Get the filename for the plot title (ICOS or local upload)
        icos_name = _icos_name.get()
        if icos_name:
            file_name = Path(icos_name).stem
        else:
            file_info = input.data_file()
            file_name = Path(file_info[0].get("name", "")).stem if file_info else ""
        axis_labels: dict[int, set[str]] = {1: set(), 2: set(), 3: set(), 4: set()}
        used_axes: set[int] = set()
        trace_count = 0
        bar_offset = 0  # counter for offsetgroup so grouped bars don't overlap

        for slot in range(1, MAX_SERIES + 1):
            s = params["series"][slot - 1]
            value_col = s["col"]
            if not value_col:
                continue
            if value_col not in df.columns:
                continue

            agg = s["agg"]
            chart = s["chart"]
            dash = s["dash"]
            color = s["color"] or DEFAULT_COLORS[slot - 1]

            try:
                axis_num = int(s["yaxis"])
            except Exception:
                axis_num = 1
            axis_num = max(1, min(4, axis_num))

            # QC filtering: if a companion column <value_col>_QC exists,
            # set values to NaN where the QC value is 2 or 3 (creates gaps in line plots).
            qc_col = f"{value_col}_QC"
            if qc_col in df.columns:
                series_df = df.copy()
                series_df.loc[series_df[qc_col].isin([2, 3]), value_col] = float("nan")
            else:
                series_df = df

            data = aggregate_series(series_df, dt_col, value_col, agg)
            if data.empty:
                continue

            yref = "y" if axis_num == 1 else f"y{axis_num}"
            name = f"{value_col} ({agg})"

            if is_profile:
                # ---- Profile view ----
                x_vals, y_vals, x_label = profile_series(series_df, dt_col, value_col, view_mode)
                if y_vals.empty:
                    continue
                name = f"{value_col} (profile)"
                # Convert x to list of strings for consistent category matching
                x_cat = [str(v) for v in x_vals.values]
                if chart == "bar":
                    fig.add_bar(
                        x=x_cat,
                        y=y_vals.values,
                        name=name,
                        marker_color=color,
                        yaxis=yref,
                        opacity=0.85,
                        offsetgroup=str(bar_offset),
                    )
                    bar_offset += 1
                else:
                    fig.add_scatter(
                        x=x_cat,
                        y=y_vals.values,
                        mode="lines+markers",
                        name=name,
                        yaxis=yref,
                        line={"color": color, "dash": dash, "width": 2},
                    )
            elif agg == "monthly" and chart == "bar":
                fig.add_bar(
                    x=data[dt_col],
                    y=data[value_col],
                    name=name,
                    marker_color=color,
                    yaxis=yref,
                    opacity=0.85,
                    offsetgroup=str(bar_offset),
                )
                bar_offset += 1
            else:
                fig.add_scatter(
                    x=data[dt_col],
                    y=data[value_col],
                    mode="lines",
                    name=name,
                    yaxis=yref,
                    line={"color": color, "dash": dash, "width": 2},
                    connectgaps=False,
                )

            axis_labels[axis_num].add(value_col)
            used_axes.add(axis_num)
            trace_count += 1

        xaxis_config = {"title": dt_col, "showline": True, "mirror": True, "ticks": "outside", "ticklen": 4, "linewidth": 1, "linecolor": "black"}
        if is_profile:
            # Use a category axis for profile views
            x_label_map = {
                "tod_30": "Time of day",
                "tod_60": "Time of day",
                "week": "Week of year",
                "month": "Month",
            }
            xaxis_config = {"title": x_label_map.get(view_mode, ""), "type": "category", "showline": True, "mirror": True, "ticks": "outside", "ticklen": 4, "linewidth": 1, "linecolor": "black"}
        elif not input.x_auto():
            x_min = parse_opt_datetime(params["x_min"])
            x_max = parse_opt_datetime(params["x_max"])
            rng = []
            if x_min is not None and not pd.isna(x_min):
                rng.append(x_min)
            else:
                rng.append(None)
            if x_max is not None and not pd.isna(x_max):
                rng.append(x_max)
            else:
                rng.append(None)
            if any(v is not None for v in rng):
                xaxis_config["range"] = rng

        # Determine plot domain to leave room for extra y-axes
        left_margin = 0.0
        right_margin = 1.0
        if 3 in used_axes:
            left_margin = 0.08  # reserve space for 2nd left axis
        if 4 in used_axes:
            right_margin = 0.92  # reserve space for 2nd right axis

        xaxis_config["domain"] = [left_margin, right_margin]

        plot_title = file_name or "Time Series"
        if is_profile:
            mode_label = VIEW_MODES.get(view_mode, "")
            plot_title = f"{file_name} — {mode_label}" if file_name else mode_label

        layout_update = {
            "template": "plotly_white",
            "title": plot_title,
            "xaxis": xaxis_config,
            "barmode": "group",
            "legend": {"orientation": "h", "yanchor": "top", "y": -0.15, "x": 0},
            "margin": {"l": 90, "r": 110, "t": 60, "b": 80},
            "autosize": True,
        }

        def _yaxis_range(ax_num: int) -> dict:
            """Return range dict for a y-axis if manual min/max are set."""
            ya = params["yaxes"][ax_num]
            if ya["auto"]:
                return {}
            rng = [None, None]
            try:
                v = ya["min"].strip()
                if v:
                    rng[0] = float(v)
            except (ValueError, AttributeError):
                pass
            try:
                v = ya["max"].strip()
                if v:
                    rng[1] = float(v)
            except (ValueError, AttributeError):
                pass
            if rng[0] is not None or rng[1] is not None:
                return {"range": rng}
            return {}

        if 1 in used_axes or trace_count == 0:
            layout_update["yaxis"] = {
                "title": ", ".join(sorted(axis_labels[1])) or "Axis 1",
                "side": "left",
                "showgrid": True,
                "showline": True, "mirror": True, "ticks": "outside", "ticklen": 4, "linewidth": 1, "linecolor": "black",
                **_yaxis_range(1),
            }

        if 2 in used_axes:
            layout_update["yaxis2"] = {
                "title": ", ".join(sorted(axis_labels[2])) or "Axis 2",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "showline": True, "ticks": "outside", "ticklen": 4, "linewidth": 1, "linecolor": "black", "linecolor": "black",
                **_yaxis_range(2),
            }

        if 3 in used_axes:
            layout_update["yaxis3"] = {
                "title": ", ".join(sorted(axis_labels[3])) or "Axis 3",
                "overlaying": "y",
                "side": "left",
                "anchor": "free",
                "position": 0.0,
                "showgrid": False,
                "autoshift": True,
                "shift": -1,
                "showline": True, "ticks": "outside", "ticklen": 4, "linewidth": 1, "linecolor": "black", "linecolor": "black",
                **_yaxis_range(3),
            }

        if 4 in used_axes:
            layout_update["yaxis4"] = {
                "title": ", ".join(sorted(axis_labels[4])) or "Axis 4",
                "overlaying": "y",
                "side": "right",
                "anchor": "free",
                "position": 1.0,
                "showgrid": False,
                "autoshift": True,
                "shift": 1,
                "showline": True, "ticks": "outside", "ticklen": 4, "linewidth": 1, "linecolor": "black",
                **_yaxis_range(4),
            }

        fig.update_layout(**layout_update)

        if trace_count == 0:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Select at least one numeric column in the series panels.",
                showarrow=False,
            )

        _current_fig.set(fig)
        return fig


    @render.download(filename=lambda: "plot.pdf")
    def download_pdf():
        fig = _current_fig.get()
        if fig is None:
            return
        buf = io.BytesIO()
        fig.write_image(buf, format="pdf", width=1200, height=700, scale=2)
        buf.seek(0)
        yield buf.read()


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")
