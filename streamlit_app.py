from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re
from typing import Callable

import altair as alt
from gspread import WorksheetNotFound
from gspread.worksheet import Worksheet
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection


SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "stockvax_sample.csv"
BASE_COLUMNS = [
    "id",
    "brand_name",
    "generic_name",
    "expiry_date",
    "stock_level",
    "expected_monthly_use",
    "order_level",
]
TEXT_COLUMNS = ["id", "brand_name", "generic_name", "expiry_date"]
NUMBER_COLUMNS = ["stock_level", "expected_monthly_use", "order_level"]
EXPIRY_PATTERN = re.compile(r"^\d{4}$")
LOG_WORKSHEET_NAME = "log"
CONSUMABLES_WORKSHEET_NAME = "consumables"
CONSUMABLES_LOG_WORKSHEET_NAME = "consumables_log"
APP_TIMEZONE = "Europe/London"
VACCINE_STOCK_COLORS = ("#ec5d4b", "#0c1722")
CONSUMABLE_STOCK_COLORS = ("#d849aa", "#0e1721")
VACCINE_ACTIVITY_COLORS = ("#ec5d4b", "#f3a43b")
CONSUMABLE_ACTIVITY_COLORS = ("#16867a", "#6fb7df")


@dataclass
class StockSnapshot:
    data: pd.DataFrame
    source_kind: str
    message: str


st.set_page_config(page_title="StockVax", layout="wide")
st.logo("data/logo.png", size="large")

def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalized.strip("_")


@st.cache_data
def load_sample_stock() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH, dtype={"expiry_date": "string"})


def empty_inventory_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=BASE_COLUMNS)


def row_has_content(row: pd.Series) -> bool:
    for column, value in row.items():
        if column == "id":
            continue
        if pd.isna(value):
            continue
        if str(value).strip() not in {"", "nan", "None"}:
            return True
    return False


def normalize_expiry_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].replace(".", "", 1).isdigit():
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if len(digits) == 3:
        digits = f"0{digits}"
    if len(digits) == 4:
        month = int(digits[:2])
        if 1 <= month <= 12:
            return digits
    return text


def make_unique_ids(values: pd.Series, blank_id_template: str) -> pd.Series:
    generated: list[str] = []
    used: set[str] = set()

    for index, raw_value in enumerate(values, start=1):
        candidate = str(raw_value).strip()
        if candidate.lower() in {"", "nan", "none"}:
            candidate = blank_id_template.format(index=index)

        original = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{original}-{suffix}"
            suffix += 1

        used.add(candidate)
        generated.append(candidate)

    return pd.Series(generated, index=values.index)


def clean_inventory_data(raw_df: pd.DataFrame | None, *, blank_id_template: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return empty_inventory_frame()

    df = raw_df.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(how="all")

    if "generic_name_description" in df.columns:
        if "generic_name" not in df.columns:
            df["generic_name"] = df["generic_name_description"]
        df = df.drop(columns=["generic_name_description"])

    if not df.empty:
        df = df[df.apply(row_has_content, axis=1)].copy()

    for column in BASE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["expiry_date"] = df["expiry_date"].map(normalize_expiry_text)

    for column in NUMBER_COLUMNS:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .round()
            .astype(int)
        )

    df["id"] = make_unique_ids(df["id"], blank_id_template=blank_id_template)

    ordered_columns = BASE_COLUMNS + [column for column in df.columns if column not in BASE_COLUMNS]
    return df[ordered_columns].reset_index(drop=True)


def clean_stock_data(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    return clean_inventory_data(raw_df, blank_id_template="VAX-{index:03d}")


def clean_consumables_data(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    return clean_inventory_data(raw_df, blank_id_template="CON-{index:03d}")


def prepare_inventory_sheet_data(
    df: pd.DataFrame,
    *,
    generic_column_name: str,
) -> pd.DataFrame:
    export_df = df.copy()
    if generic_column_name != "generic_name":
        export_df[generic_column_name] = export_df.get("generic_name", "")
        export_df = export_df.drop(columns=["generic_name"], errors="ignore")
        base_columns = [
            "id",
            "brand_name",
            generic_column_name,
            "expiry_date",
            "stock_level",
            "expected_monthly_use",
            "order_level",
        ]
    else:
        export_df = export_df.drop(columns=["generic_name_description"], errors="ignore")
        base_columns = BASE_COLUMNS

    ordered_columns = base_columns + [column for column in export_df.columns if column not in base_columns]
    return export_df[ordered_columns]


def parse_expiry_date(value: object) -> pd.Timestamp | pd.NaT:
    expiry = normalize_expiry_text(value)
    if not EXPIRY_PATTERN.fullmatch(expiry):
        return pd.NaT

    month = int(expiry[:2])
    year = 2000 + int(expiry[2:])
    return pd.Period(f"{year}-{month:02d}", freq="M").end_time.normalize()


def format_expiry_display(value: object) -> str:
    expiry = normalize_expiry_text(value)
    if not EXPIRY_PATTERN.fullmatch(expiry):
        return ""
    return f"{expiry[:2]}/{expiry[2:]}"


def status_for_row(row: pd.Series) -> str:
    days_until_expiry = row["days_until_expiry"]
    stock_level = int(row["stock_level"])
    order_level = int(row["order_level"])
    expected_monthly_use = int(row["expected_monthly_use"])

    if pd.notna(days_until_expiry) and days_until_expiry < 0 and stock_level > 0:
        return "Expired stock"
    if pd.notna(days_until_expiry) and days_until_expiry <= 90 and stock_level > 0:
        return "Expiring soon"
    if stock_level <= 0 and expected_monthly_use > 0:
        return "Out of stock"
    if order_level > 0 and stock_level <= order_level:
        return "Reorder"
    if expected_monthly_use == 0:
        return "No planned use"
    return "OK"


def add_inventory_metrics(
    stock_df: pd.DataFrame,
    *,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame] = clean_stock_data,
) -> pd.DataFrame:
    df = cleaner(stock_df)
    if df.empty:
        df["expiry_dt"] = pd.Series(dtype="datetime64[ns]")
        df["days_until_expiry"] = pd.Series(dtype="Int64")
        df["months_of_cover"] = pd.Series(dtype="float")
        df["suggested_order_qty"] = pd.Series(dtype="int")
        df["status"] = pd.Series(dtype="string")
        return df

    today = pd.Timestamp.today().normalize()
    expiry_series = pd.Series(
        [parse_expiry_date(value) for value in df["expiry_date"].tolist()],
        index=df.index,
    )
    df["expiry_dt"] = pd.to_datetime(expiry_series, errors="coerce")
    df["days_until_expiry"] = ((df["expiry_dt"] - today).dt.days).astype("Int64")

    expected_use = df["expected_monthly_use"].astype(float)
    stock_level = df["stock_level"].astype(float)
    df["months_of_cover"] = np.where(
        expected_use > 0,
        np.round(stock_level / expected_use, 1),
        np.nan,
    )
    df["suggested_order_qty"] = (
        (df["expected_monthly_use"] - df["stock_level"])
        .clip(lower=0)
        .astype(int)
    )
    df["status"] = df.apply(status_for_row, axis=1)
    return df


def selected_worksheet_name() -> str | None:
    worksheet_name = st.session_state.get("worksheet_name", "").strip()
    return worksheet_name or None


def select_live_worksheet(conn: GSheetsConnection, worksheet_name: str | None) -> Worksheet:
    return conn.client._select_worksheet(worksheet=worksheet_name)


def worksheet_column_index(worksheet: Worksheet, column_name: str) -> int:
    normalized_headers = [normalize_column_name(value) for value in worksheet.row_values(1)]
    try:
        return normalized_headers.index(column_name) + 1
    except ValueError as exc:
        raise KeyError(column_name) from exc


def worksheet_row_by_cell_value(worksheet: Worksheet, column_index: int, target_value: str) -> int | None:
    for row_number, value in enumerate(worksheet.col_values(column_index)[1:], start=2):
        if str(value).strip() == target_value:
            return row_number
    return None


def append_vaccine_log_row(
    log_worksheet: Worksheet,
    *,
    vaccine_id: str,
    vaccine_name: str,
    stock_left_before_click: int,
) -> None:
    timestamp = pd.Timestamp.now(tz=APP_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    log_worksheet.append_row(
        [
            vaccine_id,
            vaccine_name,
            stock_left_before_click,
            1,
            timestamp,
        ],
        value_input_option="USER_ENTERED",
    )


def normalized_log_header_values(values: list[object]) -> list[str]:
    return [normalize_column_name(value) for value in values[:5]]


def first_log_row_is_header(log_df: pd.DataFrame) -> bool:
    if log_df.empty:
        return False

    header_candidates = normalized_log_header_values(log_df.iloc[0].tolist())
    expected_columns = [
        {"id", "cons_id", "vaccine_id"},
        {"brand_name"},
        {"stock_level", "stock_before"},
        {"dose_given", "doses_given"},
        {"date_time", "timestamp", "datetime"},
    ]
    return len(header_candidates) >= 5 and all(
        header_candidates[index] in allowed_values
        for index, allowed_values in enumerate(expected_columns)
    )


def load_inventory_log_data(conn: GSheetsConnection, log_worksheet_name: str) -> pd.DataFrame:
    try:
        log_df = conn.read(worksheet=log_worksheet_name, ttl=0, header=None)
    except WorksheetNotFound:
        return pd.DataFrame(columns=["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"])
    except Exception:
        return pd.DataFrame(columns=["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"])

    if log_df is None or log_df.empty:
        return pd.DataFrame(columns=["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"])

    trimmed = log_df.iloc[:, :5].copy()
    while trimmed.shape[1] < 5:
        trimmed[trimmed.shape[1]] = pd.NA

    trimmed.columns = ["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"]
    if first_log_row_is_header(trimmed):
        trimmed = trimmed.iloc[1:].copy()
    trimmed = trimmed.dropna(how="all")

    for column in ["vaccine_id", "brand_name", "timestamp"]:
        trimmed[column] = trimmed[column].fillna("").astype(str).str.strip()

    trimmed["stock_before"] = pd.to_numeric(trimmed["stock_before"], errors="coerce")
    trimmed["doses_given"] = pd.to_numeric(trimmed["doses_given"], errors="coerce").fillna(1)
    trimmed["timestamp"] = pd.to_datetime(trimmed["timestamp"], errors="coerce")
    trimmed = trimmed[trimmed["timestamp"].notna()].copy()

    return trimmed.reset_index(drop=True)


def aggregate_vaccine_log_by_day(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame(columns=["day", "doses_given", "rolling_average"])

    daily_df = (
        log_df.assign(day=log_df["timestamp"].dt.floor("D"))
        .groupby("day", as_index=False)["doses_given"]
        .sum()
        .sort_values("day")
        .reset_index(drop=True)
    )
    daily_df["rolling_average"] = daily_df["doses_given"].rolling(window=7, min_periods=1).mean()
    return daily_df


def state_key(section_name: str, suffix: str) -> str:
    return f"{section_name}_{suffix}"


def apply_inventory_snapshot(
    section_name: str,
    snapshot: StockSnapshot,
    flash_kind: str | None = None,
    flash_message: str | None = None,
) -> None:
    st.session_state[state_key(section_name, "data")] = snapshot.data
    st.session_state[state_key(section_name, "source_kind")] = snapshot.source_kind
    st.session_state[state_key(section_name, "source_message")] = snapshot.message
    st.session_state[state_key(section_name, "loaded_at")] = pd.Timestamp.now()
    st.session_state.pop(state_key(section_name, "stock_editor"), None)

    if section_name == "vaccine":
        st.session_state["stock_data"] = snapshot.data
        st.session_state["source_kind"] = snapshot.source_kind
        st.session_state["source_message"] = snapshot.message
        st.session_state["loaded_at"] = pd.Timestamp.now()
        st.session_state.pop("stock_editor", None)

    if flash_kind and flash_message:
        set_flash(flash_kind, flash_message)


def fetch_inventory_snapshot(
    conn: GSheetsConnection,
    worksheet_name: str | None,
    *,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    empty_message: str,
    fallback_loader: Callable[[], pd.DataFrame] | None = None,
) -> StockSnapshot:
    try:
        df = cleaner(conn.read(worksheet=worksheet_name, ttl=0))
        if df.empty:
            fallback_data = cleaner(fallback_loader()) if fallback_loader else cleaner(None)
            return StockSnapshot(
                data=fallback_data,
                source_kind="sample" if fallback_loader else "info",
                message=empty_message,
            )
        return StockSnapshot(
            data=df,
            source_kind="google",
            message=":material/database_upload: Live data loaded from Google Sheets.",
        )
    except Exception as exc:
        fallback_data = cleaner(fallback_loader()) if fallback_loader else cleaner(None)
        return StockSnapshot(
            data=fallback_data,
            source_kind="sample" if fallback_loader else "warning",
            message=(
                f"Google Sheets could not be read ({type(exc).__name__}). Showing the bundled sample dataset instead."
                if fallback_loader
                else f"Google Sheets could not be read ({type(exc).__name__})."
            ),
        )


def fetch_snapshot(conn: GSheetsConnection, worksheet_name: str | None) -> StockSnapshot:
    return fetch_inventory_snapshot(
        conn,
        worksheet_name,
        cleaner=clean_stock_data,
        empty_message="The Google Sheet is empty right now. Sample data has been loaded locally until you save.",
        fallback_loader=load_sample_stock,
    )


def fetch_consumables_snapshot(conn: GSheetsConnection) -> StockSnapshot:
    return fetch_inventory_snapshot(
        conn,
        CONSUMABLES_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        empty_message="The consumables sheet is empty right now.",
        fallback_loader=None,
    )


def apply_snapshot(snapshot: StockSnapshot, flash_kind: str | None = None, flash_message: str | None = None) -> None:
    apply_inventory_snapshot("vaccine", snapshot, flash_kind=flash_kind, flash_message=flash_message)


def apply_consumables_snapshot(
    snapshot: StockSnapshot,
    flash_kind: str | None = None,
    flash_message: str | None = None,
) -> None:
    apply_inventory_snapshot("consumables", snapshot, flash_kind=flash_kind, flash_message=flash_message)


def set_flash(kind: str, message: str, icon: str | None = None) -> None:
    st.session_state["flash"] = {"kind": kind, "message": message, "icon": icon}


def initialize_app_state(conn: GSheetsConnection) -> None:
    if "worksheet_name" not in st.session_state:
        st.session_state["worksheet_name"] = ""

    if state_key("vaccine", "data") not in st.session_state:
        apply_snapshot(fetch_snapshot(conn, selected_worksheet_name()))
    if state_key("consumables", "data") not in st.session_state:
        apply_consumables_snapshot(fetch_consumables_snapshot(conn))


def show_flash_message() -> None:
    flash = st.session_state.pop("flash", None)
    if not flash:
        return
    getattr(st, flash["kind"])(flash["message"], icon=flash.get("icon"))


def frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        left.fillna("").reset_index(drop=True).equals(
            right.fillna("").reset_index(drop=True)
        )
    )


def normalized_sort_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def sort_inventory_cards(stock_df: pd.DataFrame, *, sort_by_brand_name: bool) -> pd.DataFrame:
    sort_df = stock_df.copy()
    sort_df["_brand_sort"] = sort_df["brand_name"].map(normalized_sort_text)
    sort_df["_generic_sort"] = sort_df["generic_name"].map(normalized_sort_text)
    sort_df["_id_sort"] = sort_df["id"].map(normalized_sort_text)

    primary_sort = "_brand_sort" if sort_by_brand_name else "_generic_sort"
    secondary_sort = "_generic_sort" if sort_by_brand_name else "_brand_sort"

    sorted_df = sort_df.sort_values(
        [primary_sort, secondary_sort, "_id_sort"],
        kind="mergesort",
    ).reset_index(drop=True)
    return sorted_df.drop(columns=["_brand_sort", "_generic_sort", "_id_sort"])


def filter_inventory_cards(stock_df: pd.DataFrame, query: str) -> pd.DataFrame:
    normalized_query = query.strip().casefold()
    if len(normalized_query) < 2:
        return stock_df

    match_mask = (
        stock_df["id"].map(normalized_sort_text).str.contains(normalized_query, regex=False)
        | stock_df["brand_name"].map(normalized_sort_text).str.contains(normalized_query, regex=False)
        | stock_df["generic_name"].map(normalized_sort_text).str.contains(normalized_query, regex=False)
    )
    return stock_df[match_mask].reset_index(drop=True)


def source_banner() -> None:
    sections = [
        ("Vaccines", st.session_state.get(state_key("vaccine", "source_kind")), st.session_state.get(state_key("vaccine", "source_message"), "")),
        ("Consumables", st.session_state.get(state_key("consumables", "source_kind")), st.session_state.get(state_key("consumables", "source_message"), "")),
    ]
    for label, source_kind, message in sections:
        if not message:
            continue
        if source_kind == "google":
            st.success(f"{label}: {message}")
        elif source_kind == "sample":
            st.warning(f"{label}: {message}")
        else:
            st.info(f"{label}: {message}")


def hex_to_rgba(value: str, alpha: float) -> str:
    hex_value = value.strip().lstrip("#")
    if len(hex_value) != 6:
        return value
    red = int(hex_value[0:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def render_title_banner() -> None:
    primary_color = st.get_option("theme.primaryColor") or "#ec5d4b"
    banner_glow_color = "#f4f6f2"
    banner_text_color = "#f4f6f2"

    banner_html = """
        <style>
        .block-container {{
            padding-top: 4.75rem;
            padding-bottom: 2.2rem;
        }}
        [data-testid="stAppViewContainer"] .main .block-container {{
            padding-top: 4.75rem;
        }}
        .hero-card {{
            background:
                radial-gradient(circle at top right, {hero_glow}, transparent 32%),
                linear-gradient(135deg, #0c1722 0%, #12202e 62%, #1f3141 100%);
            border-radius: 24px;
            padding: 1.25rem 1.6rem 1rem 1.6rem;
            color: {banner_text_color};
            box-shadow: 0 18px 40px rgba(12, 23, 34, 0.18);
            margin-bottom: 1rem;
            margin-top: 0.5rem;
            overflow: visible;
            border: 1px solid rgba(243, 248, 243, 0.18);
            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease;
        }}
        .hero-card:hover {{
            transform: translateY(8px);
            box-shadow: 0 2px 0 rgba(12, 23, 34, 0.08);
            border-color: {primary_color};
        }}
        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {banner_text_color};
            margin-bottom: 0.4rem;
            font-family: "Inter", sans-serif;
            font-weight: 800;
        }}
        .hero-title {{
            margin: 0;
            font-family: "Inter", sans-serif;
            font-size: 2.8rem;
            font-weight: 900;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin-bottom: 0.55rem;
            color: {banner_text_color};
        }}
        .hero-subtitle {{
            margin: 0;
            font-size: 0.85rem;
            max-width: 58rem;
            color: {banner_text_color};
            line-height: 1.45;
            font-family: "Inter", sans-serif;
            font-weight: 400;
        }}
        @media (max-width: 640px) {{
            .hero-card {{
                padding: 1rem 1rem 0.9rem 1rem;
            }}
            .hero-kicker {{
                font-size: 0.72rem;
            }}
            .hero-title {{
                font-size: 2rem;
                margin-bottom: 0.45rem;
            }}
            .hero-subtitle {{
                font-size: 0.88rem;
            }}
        }}
        </style>

        <section class="hero-card">
            <div class="hero-kicker">Stanhope Mews Surgery</div>
            <h1 class="hero-title">StockVax</h1>
            <p class="hero-subtitle">
                Live surgery vaccine and consumables inventory powered by Google Sheets & Streamlit, with clear reorder and expiry visibility. Code by janduplessis883.
            </p>
        </section>
        """.format(
        primary_color=primary_color,
        banner_text_color=banner_text_color,
        hero_glow=hex_to_rgba(banner_glow_color, 0.18),
    )
    st.html(banner_html)


def render_sidebar(conn: GSheetsConnection) -> None:
    st.sidebar.header("Sheet connection")
    st.sidebar.text_input(
        "Vaccine worksheet name",
        key="worksheet_name",
        help="Leave blank to use the first vaccine worksheet tab in the spreadsheet from .streamlit/secrets.toml. Consumables always use the 'consumables' tab.",
        placeholder="First worksheet tab",
    )

    refresh_clicked = st.sidebar.button("Refresh from Google Sheet", width="stretch")
    sample_clicked = st.sidebar.button("Load sample data locally", width="stretch", disabled=True)

    st.sidebar.caption("Editor saves rewrite the relevant table in Google Sheets, so avoid concurrent edits in multiple browser sessions.")

    if refresh_clicked:
        apply_snapshot(
            fetch_snapshot(conn, selected_worksheet_name()),
        )
        apply_consumables_snapshot(fetch_consumables_snapshot(conn))
        set_flash("info", ":material/refresh: Vaccine and consumables data refreshed from Google Sheets.")
        st.rerun()

    if sample_clicked:
        apply_snapshot(
            StockSnapshot(
                data=clean_stock_data(load_sample_stock()),
                source_kind="sample",
                message="Sample data is loaded locally. Nothing has been written back to Google Sheets yet.",
            ),
            flash_kind="info",
            flash_message="Sample stock data loaded into the editor.",
        )
        st.rerun()


def render_metrics(
    stock_df: pd.DataFrame,
    *,
    tracked_label: str,
    on_hand_label: str,
) -> None:
    total_stock = int(stock_df["stock_level"].sum())
    low_stock_count = int(stock_df["status"].isin(["Reorder", "Out of stock"]).sum())
    expiring_count = int(
        ((stock_df["days_until_expiry"] <= 90) & (stock_df["days_until_expiry"] >= 0) & (stock_df["stock_level"] > 0)).sum()
    )
    months_of_cover = (
        round(total_stock / stock_df["expected_monthly_use"].sum(), 1)
        if int(stock_df["expected_monthly_use"].sum()) > 0
        else 0
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(tracked_label, len(stock_df))
    metric_columns[1].metric(on_hand_label, total_stock)
    metric_columns[2].metric("Need action", low_stock_count)
    metric_columns[3].metric("Months of cover", months_of_cover)

    st.caption(f"Expiring in the next 90 days: {expiring_count}")


def watchlist_row_styles(row: pd.Series, table_kind: str) -> list[str]:
    text_color = st.get_option("theme.textColor") or "#0c1722"
    primary_color = st.get_option("theme.primaryColor") or "#ec5d4b"
    red_color = st.get_option("theme.redColor") or primary_color
    reorder_color = "#f4f6f2"
    expiry_soon_color = "#e4af6c"

    background = ""
    border = ""
    status = row.get("status", "")

    if status == "Out of stock":
        background = hex_to_rgba(red_color, 0.14)
        border = f"4px solid {red_color}"
    elif status == "Expired stock":
        background = hex_to_rgba(red_color, 0.12)
        border = f"4px solid {red_color}"
    elif status == "Reorder":
        background = reorder_color
        border = f"4px solid {primary_color}"
    elif status == "Expiring soon":
        background = hex_to_rgba(expiry_soon_color, 0.22)
        border = f"4px solid {expiry_soon_color}"
    elif table_kind == "expiry" and pd.notna(row.get("days_until_expiry")) and int(row.get("days_until_expiry", 9999)) <= 90:
        background = hex_to_rgba(expiry_soon_color, 0.18)
        border = f"4px solid {expiry_soon_color}"

    base_style = f"color: {text_color};"
    if not background:
        return [base_style for _ in row]

    styles: list[str] = []
    for column in row.index:
        cell_style = f"{base_style} background-color: {background};"
        if column == row.index[0]:
            cell_style += f" border-left: {border};"
        styles.append(cell_style)
    return styles


def styled_watchlist_table(df: pd.DataFrame, table_kind: str) -> pd.io.formats.style.Styler:
    styled = df.style.apply(
        lambda row: watchlist_row_styles(row, table_kind),
        axis=1,
    )
    if "months_of_cover" in df.columns:
        styled = styled.format(
            {
                "months_of_cover": lambda value: ""
                if pd.isna(value)
                else f"{float(value):.1f}"
            }
        )
    return styled


def render_stock_usage_plot(
    stock_df: pd.DataFrame,
    *,
    title: str,
    series_colors: tuple[str, str],
    empty_message: str,
    item_label: str,
    value_label: str,
) -> None:
    plot_df = (
        stock_df[["brand_name", "stock_level", "expected_monthly_use"]]
        .sort_values("brand_name")
        .reset_index(drop=True)
    )
    if plot_df.empty:
        st.info(empty_message)
        return

    primary_color, comparison_color = series_colors
    text_color = st.get_option("theme.textColor") or "#0c1722"
    grid_color = st.get_option("theme.borderColor") or "#d7ded8"
    tick_color = "#7d8386"
    long_plot_df = plot_df.rename(
        columns={
            "stock_level": "Stock level",
            "expected_monthly_use": "Expected monthly use",
        }
    ).melt(
        id_vars="brand_name",
        value_vars=["Stock level", "Expected monthly use"],
        var_name="series",
        value_name="value",
    )
    long_plot_df["series_order"] = long_plot_df["series"].map(
        {
            "Stock level": 0,
            "Expected monthly use": 1,
        }
    )

    chart = (
        alt.Chart(long_plot_df)
        .mark_bar(size=46)
        .encode(
            x=alt.X(
                "brand_name:N",
                sort=plot_df["brand_name"].tolist(),
                axis=alt.Axis(
                    title=None,
                    labelAngle=-90,
                    labelColor=tick_color,
                    labelFontSize=11,
                    labelPadding=10,
                    ticks=False,
                ),
            ),
            y=alt.Y(
                "sum(value):Q",
                title=None,
                axis=alt.Axis(
                    labelColor=tick_color,
                    labelFontSize=11,
                    gridColor=grid_color,
                    domain=False,
                    tickColor=grid_color,
                ),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Stock level", "Expected monthly use"],
                    range=[primary_color, comparison_color],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    direction="horizontal",
                    labelColor=tick_color,
                    symbolType="square",
                ),
            ),
            order=alt.Order(
                "series_order:Q",
                sort="ascending",
            ),
            tooltip=[
                alt.Tooltip("brand_name:N", title=item_label),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=value_label, format=".0f"),
            ],
        )
        .properties(
            title=title,
            height=420,
        )
        .configure_title(
            anchor="start",
            color=text_color,
            fontSize=18,
            fontWeight="bold",
        )
        .configure_view(strokeOpacity=0)
    )

    st.altair_chart(chart, width="stretch")


def render_inventory_activity_plot(
    log_df: pd.DataFrame,
    *,
    title: str,
    empty_message: str,
    item_label_plural: str,
    activity_value_label: str,
    activity_colors: tuple[str, str],
) -> None:
    daily_df = aggregate_vaccine_log_by_day(log_df)
    if daily_df.empty:
        st.info(empty_message)
        return

    primary_color, accent_color = activity_colors
    text_color = st.get_option("theme.textColor") or "#0c1722"
    grid_color = st.get_option("theme.borderColor") or "#d7ded8"
    tick_color = "#7d8386"

    peak_day = daily_df.loc[daily_df["doses_given"].idxmax()]
    total_doses = int(daily_df["doses_given"].sum())
    active_days = int((daily_df["doses_given"] > 0).sum())

    base = alt.Chart(daily_df).encode(
        x=alt.X(
            "day:T",
            title=None,
            axis=alt.Axis(
                format="%d %b",
                labelColor=tick_color,
                labelFontSize=11,
                grid=False,
                domain=False,
                tickColor=grid_color,
            ),
        )
    )

    area = base.mark_area(
        line={"color": primary_color, "strokeWidth": 2.6},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color=primary_color, offset=0),
                alt.GradientStop(color="#fff4ef", offset=1),
            ],
            x1=1,
            x2=1,
            y1=1,
            y2=0,
        ),
        opacity=0.32,
    ).encode(
        y=alt.Y(
            "doses_given:Q",
            title=None,
            axis=alt.Axis(
                labelColor=tick_color,
                labelFontSize=11,
                gridColor=grid_color,
                domain=False,
                tickColor=grid_color,
            ),
        ),
        tooltip=[
            alt.Tooltip("day:T", title="Day", format="%d %B %Y"),
            alt.Tooltip("doses_given:Q", title=activity_value_label, format=".0f"),
            alt.Tooltip("rolling_average:Q", title="7-day average", format=".1f"),
        ],
    )

    pulse_points = base.mark_circle(
        color=text_color,
        opacity=0.9,
        stroke="white",
        strokeWidth=1.2,
    ).encode(
        y="doses_given:Q",
        size=alt.Size("doses_given:Q", scale=alt.Scale(range=[70, 430]), legend=None),
    )

    trend_line = base.mark_line(
        color=accent_color,
        strokeDash=[8, 6],
        strokeWidth=2.4,
    ).encode(
        y="rolling_average:Q",
        tooltip=[
            alt.Tooltip("day:T", title="Day", format="%d %B %Y"),
            alt.Tooltip("rolling_average:Q", title="7-day average", format=".1f"),
        ],
    )

    peak_highlight = (
        alt.Chart(pd.DataFrame([peak_day]))
        .mark_point(shape="diamond", filled=True, size=280, color=accent_color, stroke="white", strokeWidth=1.5)
        .encode(
            x="day:T",
            y="doses_given:Q",
            tooltip=[
                alt.Tooltip("day:T", title="Peak day", format="%d %B %Y"),
                alt.Tooltip("doses_given:Q", title=activity_value_label, format=".0f"),
            ],
        )
    )

    peak_label = (
        alt.Chart(pd.DataFrame([peak_day]))
        .mark_text(
            align="left",
            baseline="bottom",
            dx=10,
            dy=-8,
            color=text_color,
            fontSize=12,
            fontWeight="bold",
        )
        .encode(
            x="day:T",
            y="doses_given:Q",
            text=alt.value(f"Peak day: {int(peak_day['doses_given'])}"),
        )
    )

    chart = (
        alt.layer(area, trend_line, pulse_points, peak_highlight, peak_label)
        .properties(
            title=title,
            height=320,
        )
        .configure_title(
            anchor="start",
            color=text_color,
            fontSize=18,
            fontWeight="bold",
        )
        .configure_view(strokeOpacity=0)
    )

    st.altair_chart(chart, width="stretch")
    st.caption(
        f"{total_doses} {item_label_plural.lower()} logged across {active_days} day(s). "
        f"Peak activity was {int(peak_day['doses_given'])} on {peak_day['day']:%d %b %Y}."
    )


def render_inventory_overview(
    conn: GSheetsConnection,
    stock_df: pd.DataFrame,
    *,
    section_title: str,
    tracked_label: str,
    on_hand_label: str,
    item_label_plural: str,
    value_label: str,
    activity_value_label: str,
    log_worksheet_name: str,
    stock_chart_colors: tuple[str, str],
    activity_chart_colors: tuple[str, str],
) -> None:
    render_metrics(stock_df, tracked_label=tracked_label, on_hand_label=on_hand_label)

    reorder_df = stock_df[stock_df["status"].isin(["Reorder", "Out of stock"])].copy()
    reorder_df = reorder_df.sort_values(["status", "suggested_order_qty", "brand_name"], ascending=[True, False, True])

    expiry_df = stock_df[
        stock_df["days_until_expiry"].notna() & (stock_df["stock_level"] > 0)
    ].copy()
    expiry_df = expiry_df.sort_values("days_until_expiry")

    invalid_expiry_count = int(
        ((stock_df["expiry_date"] != "") & stock_df["expiry_dt"].isna()).sum()
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Reorder watchlist")
        if reorder_df.empty:
            st.info(f"No {item_label_plural.lower()} are currently at or below their order level.")
        else:
            st.dataframe(
                styled_watchlist_table(
                    reorder_df[
                        [
                            "brand_name",
                            "generic_name",
                            "stock_level",
                            "order_level",
                            "expected_monthly_use",
                            "suggested_order_qty",
                            "status",
                        ]
                    ],
                    "reorder",
                ),
                width="stretch",
                hide_index=True,
            )

    with right_column:
        st.subheader("Expiry watchlist")
        if expiry_df.empty:
            st.info(f"No {item_label_plural.lower()} have an expiry date recorded yet.")
        else:
            st.dataframe(
                styled_watchlist_table(
                    expiry_df[
                        [
                            "brand_name",
                            "generic_name",
                            "expiry_date",
                            "days_until_expiry",
                            "stock_level",
                            "status",
                        ]
                    ],
                    "expiry",
                ),
                width="stretch",
                hide_index=True,
            )

    if invalid_expiry_count:
        st.warning(f"{invalid_expiry_count} expiry value(s) could not be parsed. Use MMYY format, for example 0328.")

    render_stock_usage_plot(
        stock_df,
        title=f"{section_title}: current stock vs expected monthly use",
        series_colors=stock_chart_colors,
        empty_message=f"No {item_label_plural.lower()} data is available to plot yet.",
        item_label=section_title,
        value_label=value_label,
    )
    render_inventory_activity_plot(
        load_inventory_log_data(conn, log_worksheet_name),
        title=f"Daily {item_label_plural.lower()} activity",
        empty_message=f"No {item_label_plural.lower()} activity has been logged yet.",
        item_label_plural=item_label_plural,
        activity_value_label=activity_value_label,
        activity_colors=activity_chart_colors,
    )

    st.subheader("Current inventory")
    st.dataframe(
        styled_watchlist_table(
            stock_df[
                [
                    "id",
                    "brand_name",
                    "generic_name",
                    "expiry_date",
                    "stock_level",
                    "expected_monthly_use",
                    "order_level",
                    "months_of_cover",
                    "status",
                ]
            ],
            "inventory",
        ),
        width="stretch",
        hide_index=True,
    )


def record_inventory_click(
    conn: GSheetsConnection,
    item_id: str,
    *,
    worksheet_name: str | None,
    log_worksheet_name: str,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    snapshot_applier: Callable[[StockSnapshot], None],
    item_label_singular: str,
    action_word: str,
) -> None:
    try:
        latest_df = cleaner(conn.read(worksheet=worksheet_name, ttl=0))
        stock_worksheet = select_live_worksheet(conn, worksheet_name)
    except Exception as exc:
        set_flash(
            "error",
            f"Could not record the {item_label_singular.lower()} because Google Sheets could not be reached: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    if latest_df.empty:
        set_flash(
            "error",
            f"Could not record the {item_label_singular.lower()} because the sheet is empty.",
            icon=":material/error:",
        )
        st.rerun()

    match = latest_df["id"] == item_id
    if not match.any():
        set_flash(
            "error",
            f"That {item_label_singular.lower()} could not be found in the latest stock list. Please refresh and try again.",
            icon=":material/error:",
        )
        st.rerun()

    row_index = latest_df.index[match][0]
    item_name = latest_df.at[row_index, "brand_name"] or latest_df.at[row_index, "generic_name"] or item_id
    current_stock = int(latest_df.at[row_index, "stock_level"])

    if current_stock <= 0:
        set_flash(
            "warning",
            f"{item_name} is already at zero stock, so no entry was recorded.",
            icon=":material/warning:",
        )
        st.rerun()

    latest_df.at[row_index, "stock_level"] = current_stock - 1

    try:
        id_column_index = worksheet_column_index(stock_worksheet, "id")
        stock_column_index = worksheet_column_index(stock_worksheet, "stock_level")
        log_worksheet = select_live_worksheet(conn, log_worksheet_name)
        worksheet_row = worksheet_row_by_cell_value(stock_worksheet, id_column_index, item_id)

        if worksheet_row is None:
            raise KeyError(f"ID {item_id} was not found in the live sheet.")

        stock_worksheet.update_cell(worksheet_row, stock_column_index, current_stock - 1)
        append_vaccine_log_row(
            log_worksheet,
            vaccine_id=item_id,
            vaccine_name=item_name,
            stock_left_before_click=current_stock,
        )
    except WorksheetNotFound:
        set_flash(
            "error",
            f"Could not record the {item_label_singular.lower()} because the '{log_worksheet_name}' worksheet was not found.",
            icon=":material/error:",
        )
        st.rerun()
    except KeyError as exc:
        set_flash(
            "error",
            f"Could not record the {item_label_singular.lower()} because the live sheet structure did not match the app: {exc}",
            icon=":material/error:",
        )
        st.rerun()
    except Exception as exc:
        set_flash(
            "error",
            f"Could not save the updated stock and log entry for {item_name}: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    snapshot_applier(
        StockSnapshot(
            data=latest_df,
            source_kind="google",
            message="Live data loaded from Google Sheets.",
        ),
    )
    set_flash(
        "success",
        f"{item_name}: 1 {item_label_singular.lower()} {action_word} and recorded successfully.",
        icon=":material/check_circle:",
    )
    st.rerun()


def give_vaccine_dose(conn: GSheetsConnection, vaccine_id: str) -> None:
    record_inventory_click(
        conn,
        vaccine_id,
        worksheet_name=selected_worksheet_name(),
        log_worksheet_name=LOG_WORKSHEET_NAME,
        cleaner=clean_stock_data,
        snapshot_applier=apply_snapshot,
        item_label_singular="Vaccine",
        action_word="given",
    )


def use_consumable(conn: GSheetsConnection, consumable_id: str) -> None:
    record_inventory_click(
        conn,
        consumable_id,
        worksheet_name=CONSUMABLES_WORKSHEET_NAME,
        log_worksheet_name=CONSUMABLES_LOG_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        snapshot_applier=apply_consumables_snapshot,
        item_label_singular="Consumable",
        action_word="used",
    )


def render_inventory_action_tab(
    conn: GSheetsConnection,
    stock_df: pd.DataFrame,
    *,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    expander_title: str,
    section_subheader: str,
    chart_item_label: str,
    caption: str,
    empty_message: str,
    button_label: str,
    button_icon: str,
    button_key_prefix: str,
    on_click: Callable[[GSheetsConnection, str], None],
    chart_colors: tuple[str, str],
    brand_font_size: str = "2.0rem",
    generic_font_size: str = "0.95rem",
    expiry_font_size: str = "1.5rem",
) -> None:
    with st.expander(expander_title, icon=":material/bar_chart:"):
        render_stock_usage_plot(
            add_inventory_metrics(stock_df, cleaner=cleaner),
            title="Current stock vs expected monthly use",
            series_colors=chart_colors,
            empty_message=empty_message,
            item_label=chart_item_label,
            value_label="Units",
        )
    control_columns = st.columns([1, 1.35])
    with control_columns[0]:
        sort_by_brand_name = st.toggle(
            "Sort by Brand Name",
            value=False,
            key=f"{button_key_prefix}_sort",
            help="Off sorts the list by Generic Name.",
        )
    with control_columns[1]:
        search_query = st.text_input(
            "Search",
            key=f"{button_key_prefix}_search",
            placeholder="Search by ID, brand, or generic name",
            help="Filters by ID, brand name, or generic name / description.",
            label_visibility="collapsed",
        )
    st.subheader(section_subheader)
    st.caption(caption)
    st.html(
        f"""
        <style>
        .nurse-card-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.3rem;
        }}
        .nurse-card-title-block {{
            min-width: 0;
            flex: 1;
        }}
        .nurse-brand-name {{
            margin: 0 0 0.2rem 0;
            font-size: {brand_font_size};
            line-height: 1.05;
            font-weight: 800;
            color: #0c1722;
            letter-spacing: -0.03em;
        }}
        .nurse-generic-name {{
            margin: 0 0 0.85rem 0;
            font-size: {generic_font_size};
            line-height: 1.2;
            font-weight: 500;
            color: #6b7280;
            letter-spacing: -0.01em;
        }}
        .nurse-expiry-pill {{
            flex-shrink: 0;
            min-width: 6.4rem;
            padding: 0.4rem 0.95rem;
            border-radius: 999px;
            background: rgba(216, 222, 216, 0.32);
            border: 1px solid rgba(216, 222, 216, 0.8);
            box-shadow: 0 4px 12px rgba(12, 23, 34, 0.08);
            font-size: {expiry_font_size};
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.03em;
            text-align: center;
            color: #d8ded8;
        }}
        .nurse-expiry-pill.is-warning {{
            background: rgba(211, 159, 90, 0.16);
            border-color: rgba(211, 159, 90, 0.52);
            color: #d39f5a;
            box-shadow: 0 4px 12px rgba(211, 159, 90, 0.16);
        }}
        .nurse-expiry-pill.is-urgent {{
            background: rgba(236, 93, 75, 0.14);
            border-color: rgba(236, 93, 75, 0.52);
            color: #ec5d4b;
            box-shadow: 0 4px 12px rgba(236, 93, 75, 0.14);
        }}
        </style>
        """
    )

    display_df = sort_inventory_cards(stock_df, sort_by_brand_name=sort_by_brand_name)
    display_df = filter_inventory_cards(display_df, search_query)
    if display_df.empty:
        if len(search_query.strip()) >= 2:
            st.info("No matching items were found for that search.")
        else:
            st.info(empty_message)
        return

    card_columns = st.columns(3)

    for index, (_, row) in enumerate(display_df.iterrows()):
        with card_columns[index % 3]:
            with st.container(border=True):
                brand_name = row["brand_name"] or row["generic_name"] or "Unnamed item"
                generic_name = row["generic_name"] or "No description"
                stock_level = int(row["stock_level"])
                order_level = int(row["order_level"])
                today = pd.Timestamp.today().normalize()
                expiry_dt = parse_expiry_date(row["expiry_date"])
                expiry_display = format_expiry_display(row["expiry_date"])
                expiry_class = "nurse-expiry-pill"
                if pd.notna(expiry_dt):
                    if expiry_dt <= (today + pd.DateOffset(months=1)):
                        expiry_class = "nurse-expiry-pill is-urgent"
                    elif expiry_dt <= (today + pd.DateOffset(months=3)):
                        expiry_class = "nurse-expiry-pill is-warning"
                expiry_markup = (
                    f'<div class="{expiry_class}">{html.escape(expiry_display)}</div>'
                    if expiry_display
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="nurse-card-header">
                        <div class="nurse-card-title-block">
                            <div class="nurse-brand-name">{html.escape(brand_name)}</div>
                            <div class="nurse-generic-name">{html.escape(generic_name)}</div>
                        </div>
                        {expiry_markup}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                detail_columns = st.columns(2)
                detail_columns[0].metric("Stock", f"{stock_level}")
                detail_columns[1].metric(":gray[Order level]", f":gray[{order_level}]")

                if stock_level <= 0:
                    st.error("Out of stock", icon=":material/error:")
                elif stock_level <= order_level and order_level > 0:
                    st.warning("Low stock / Reorder", icon=":material/warning:")
                else:
                    st.info("Ready to record", icon=":material/check_circle:")

                if st.button(
                    button_label,
                    key=f"{button_key_prefix}_{row['id']}",
                    type="primary",
                    icon=button_icon,
                    disabled=stock_level <= 0,
                    width="stretch",
                ):
                    on_click(conn, row["id"])


def render_nurse_tab(conn: GSheetsConnection, stock_df: pd.DataFrame) -> None:
    render_inventory_action_tab(
        conn,
        stock_df,
        cleaner=clean_stock_data,
        expander_title="Stock usage bar plot",
        section_subheader="Nurse recording",
        chart_item_label="Vaccine",
        caption="Keep this tab open during clinic. One click records one vaccine given and updates the live stock level immediately.",
        empty_message="No vaccines are available in the stock list yet.",
        button_label="Record 1 given",
        button_icon=":material/syringe:",
        button_key_prefix="give_dose",
        on_click=give_vaccine_dose,
        chart_colors=VACCINE_STOCK_COLORS,
    )


def render_consumables_tab(conn: GSheetsConnection, stock_df: pd.DataFrame) -> None:
    render_inventory_action_tab(
        conn,
        stock_df,
        cleaner=clean_consumables_data,
        expander_title="Consumables stock usage bar plot",
        section_subheader="Consumables recording",
        chart_item_label="Consumable",
        caption="Use this tab to record one consumable used and update the live stock level immediately.",
        empty_message="No consumables are available in the stock list yet.",
        button_label="Record 1 used",
        button_icon=":material/inventory_2:",
        button_key_prefix="use_consumable",
        on_click=use_consumable,
        chart_colors=CONSUMABLE_STOCK_COLORS,
        brand_font_size="1.55rem",
        generic_font_size="0.82rem",
        expiry_font_size="1.2rem",
    )


def apply_inventory_delivery(
    conn: GSheetsConnection,
    deliveries: dict[str, int],
    *,
    worksheet_name: str | None,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    item_label_plural: str,
    generic_column_name: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    positive_deliveries = {item_id: amount for item_id, amount in deliveries.items() if amount > 0}
    if not positive_deliveries:
        return None, []

    latest_df = cleaner(conn.read(worksheet=worksheet_name, ttl=0))
    if latest_df.empty:
        raise ValueError(f"The {item_label_plural.lower()} sheet is empty.")

    recorded_items: list[str] = []
    for item_id, amount in positive_deliveries.items():
        match = latest_df["id"] == item_id
        if not match.any():
            continue

        row_index = latest_df.index[match][0]
        item_name = latest_df.at[row_index, "brand_name"] or latest_df.at[row_index, "generic_name"] or item_id
        latest_df.at[row_index, "stock_level"] = int(latest_df.at[row_index, "stock_level"]) + int(amount)
        recorded_items.append(f"{item_name} (+{amount})")

    if not recorded_items:
        raise ValueError(f"None of the {item_label_plural.lower()} delivery rows matched the latest live list.")

    conn.update(
        worksheet=worksheet_name,
        data=prepare_inventory_sheet_data(latest_df, generic_column_name=generic_column_name),
    )
    return latest_df, recorded_items


def record_delivery(
    conn: GSheetsConnection,
    vaccine_deliveries: dict[str, int],
    consumable_deliveries: dict[str, int],
) -> None:
    if not any(amount > 0 for amount in vaccine_deliveries.values()) and not any(
        amount > 0 for amount in consumable_deliveries.values()
    ):
        set_flash(
            "info",
            "No delivery quantities were entered.",
            icon=":material/info:",
        )
        st.rerun()

    summary_lines: list[str] = []
    completed_sections: list[str] = []

    try:
        vaccine_df, vaccine_items = apply_inventory_delivery(
            conn,
            vaccine_deliveries,
            worksheet_name=selected_worksheet_name(),
            cleaner=clean_stock_data,
            item_label_plural="Vaccines",
            generic_column_name="generic_name",
        )
        if vaccine_df is not None:
            apply_snapshot(
                StockSnapshot(
                    data=vaccine_df,
                    source_kind="google",
                    message="Live data loaded from Google Sheets.",
                ),
            )
            completed_sections.append("vaccines")
            summary_lines.append(f"Vaccines: {', '.join(vaccine_items[:4])}")

        consumables_df, consumables_items = apply_inventory_delivery(
            conn,
            consumable_deliveries,
            worksheet_name=CONSUMABLES_WORKSHEET_NAME,
            cleaner=clean_consumables_data,
            item_label_plural="Consumables",
            generic_column_name="generic_name_description",
        )
        if consumables_df is not None:
            apply_consumables_snapshot(
                StockSnapshot(
                    data=consumables_df,
                    source_kind="google",
                    message="Live data loaded from Google Sheets.",
                ),
            )
            completed_sections.append("consumables")
            summary_lines.append(f"Consumables: {', '.join(consumables_items[:4])}")
    except Exception as exc:
        partial_note = " Some earlier changes may already be saved." if completed_sections else ""
        set_flash(
            "error",
            f"Could not save the delivery update: {exc}.{partial_note}",
            icon=":material/error:",
        )
        st.rerun()

    set_flash(
        "success",
        "Delivery recorded successfully. " + " | ".join(summary_lines),
        icon=":material/check_circle:",
    )
    st.rerun()


def render_delivery_section(
    display_df: pd.DataFrame,
    *,
    section_title: str,
    input_key_prefix: str,
) -> None:
    st.markdown(f"#### {section_title}")
    if display_df.empty:
        st.info(f"No {section_title.lower()} are available in the stock list yet.")
        return

    for _, row in display_df.iterrows():
        brand_name = row["brand_name"] or row["generic_name"] or "Unnamed item"
        generic_name = row["generic_name"] or "No description"
        item_id = row["id"]
        expiry_display = format_expiry_display(row["expiry_date"]) or "Not set"

        info_column, stock_column, input_column = st.columns([2.3, 1, 1.1])
        with info_column:
            st.markdown(
                f"""
                <div class="delivery-name">{html.escape(brand_name)}</div>
                <div class="delivery-generic">{html.escape(generic_name)}</div>
                """,
                unsafe_allow_html=True,
            )
        with stock_column:
            st.markdown(
                f"""
                <div class="delivery-stock">Current stock: {int(row["stock_level"])}</div>
                <div class="delivery-expiry">Expiry Date: {html.escape(expiry_display)}</div>
                """,
                unsafe_allow_html=True,
            )
        with input_column:
            st.number_input(
                f"Delivery for {brand_name}",
                min_value=0,
                step=1,
                value=0,
                key=f"{input_key_prefix}_{item_id}",
                label_visibility="collapsed",
                placeholder="0",
                width="stretch",
            )


def render_delivery_tab(conn: GSheetsConnection, vaccine_df: pd.DataFrame, consumables_df: pd.DataFrame) -> None:
    st.subheader("Record delivery")
    st.caption("Enter any new stock delivered today for vaccines or consumables, then save once to add the quantities onto the live stock levels.")

    vaccine_display_df = vaccine_df.sort_values(["brand_name", "generic_name"]).reset_index(drop=True)
    consumables_display_df = consumables_df.sort_values(["brand_name", "generic_name"]).reset_index(drop=True)
    if vaccine_display_df.empty and consumables_display_df.empty:
        st.info("No vaccines or consumables are available in the stock list yet.")
        return

    st.html(
        """
        <style>
        .delivery-row {
            display: grid;
            grid-template-columns: minmax(0, 1.7fr) minmax(120px, 0.8fr) minmax(160px, 0.9fr);
            gap: 1rem;
            align-items: center;
            padding: 0.8rem 0.2rem;
            border-bottom: 1px solid rgba(215, 222, 216, 0.9);
        }
        .delivery-row:last-child {
            border-bottom: none;
        }
        .delivery-name {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0c1722;
            line-height: 1.15;
        }
        .delivery-generic {
            margin-top: 0.18rem;
            font-size: 0.88rem;
            color: #7d8386;
            line-height: 1.15;
            font-weight: 400;
        }
        .delivery-stock {
            font-size: 0.95rem;
            color: #0c1722;
            font-weight: 600;
        }
        .delivery-expiry {
            margin-top: 0.18rem;
            font-size: 0.88rem;
            color: #7d8386;
            font-weight: 400;
            line-height: 1.15;
        }
        @media (max-width: 900px) {
            .delivery-row {
                grid-template-columns: 1fr;
                gap: 0.4rem;
                padding: 0.85rem 0;
            }
        }
        </style>
        """
    )

    with st.form("record_delivery_form", clear_on_submit=True, width="stretch"):
        render_delivery_section(
            vaccine_display_df,
            section_title="Vaccines",
            input_key_prefix="delivery_vaccine",
        )
        st.divider()
        render_delivery_section(
            consumables_display_df,
            section_title="Consumables",
            input_key_prefix="delivery_consumable",
        )

        submitted = st.form_submit_button(
            "Record delivery",
            type="primary",
            icon=":material/local_shipping:",
            width="stretch",
        )

    if submitted:
        vaccine_deliveries = {
            row["id"]: int(st.session_state.get(f"delivery_vaccine_{row['id']}", 0))
            for _, row in vaccine_display_df.iterrows()
        }
        consumable_deliveries = {
            row["id"]: int(st.session_state.get(f"delivery_consumable_{row['id']}", 0))
            for _, row in consumables_display_df.iterrows()
        }
        record_delivery(conn, vaccine_deliveries, consumable_deliveries)


def render_editor_section(
    conn: GSheetsConnection,
    stock_df: pd.DataFrame,
    *,
    section_title: str,
    worksheet_name: str | None,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    snapshot_applier: Callable[[StockSnapshot, str | None, str | None], None] | Callable[[StockSnapshot], None],
    editor_key: str,
    generic_name_label: str,
    generic_column_name: str,
    save_button_label: str,
    download_file_name: str,
) -> None:
    st.markdown(f"### {section_title}")
    editor_columns = BASE_COLUMNS + [column for column in stock_df.columns if column not in BASE_COLUMNS]
    editable_df = stock_df[editor_columns].copy()

    edited_df = st.data_editor(
        editable_df,
        key=editor_key,
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        disabled=["id"],
        column_config={
            "id": st.column_config.TextColumn("ID"),
            "brand_name": st.column_config.TextColumn("Brand name", required=True),
            "generic_name": st.column_config.TextColumn(generic_name_label, required=True),
            "expiry_date": st.column_config.TextColumn("Expiry (MMYY)"),
            "stock_level": st.column_config.NumberColumn("Stock level", min_value=0, step=1, format="%d"),
            "expected_monthly_use": st.column_config.NumberColumn(
                "Expected monthly use",
                min_value=0,
                step=1,
                format="%d",
            ),
            "order_level": st.column_config.NumberColumn("Order level", min_value=0, step=1, format="%d"),
        },
    )

    cleaned_edited_df = cleaner(edited_df)
    changes_pending = not frames_match(cleaned_edited_df, cleaner(stock_df))

    if changes_pending:
        st.info(f"You have unsaved changes in the {section_title.lower()} editor.")

    button_columns = st.columns([1, 1, 1.2])
    save_clicked = button_columns[0].button(save_button_label, type="primary", width="stretch", key=f"{editor_key}_save")
    discard_clicked = button_columns[1].button("Discard local changes", width="stretch", key=f"{editor_key}_discard")
    button_columns[2].download_button(
        "Download CSV snapshot",
        data=cleaned_edited_df.to_csv(index=False),
        file_name=download_file_name,
        mime="text/csv",
        width="stretch",
        key=f"{editor_key}_download",
    )

    if save_clicked:
        try:
            conn.update(
                worksheet=worksheet_name,
                data=prepare_inventory_sheet_data(cleaned_edited_df, generic_column_name=generic_column_name),
            )
            snapshot_applier(
                StockSnapshot(
                    data=cleaned_edited_df,
                    source_kind="google",
                    message="Changes saved to Google Sheets.",
                ),
                "success",
                f"The {section_title.lower()} table was saved to Google Sheets.",
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save {section_title.lower()} to Google Sheets: {exc}")

    if discard_clicked:
        if worksheet_name == CONSUMABLES_WORKSHEET_NAME:
            snapshot = fetch_consumables_snapshot(conn)
            apply_consumables_snapshot(
                snapshot,
                flash_kind="info",
                flash_message=f"Local {section_title.lower()} changes were discarded.",
            )
        else:
            snapshot = fetch_snapshot(conn, worksheet_name)
            apply_snapshot(
                snapshot,
                flash_kind="info",
                flash_message=f"Local {section_title.lower()} changes were discarded.",
            )
        st.rerun()


def render_editor(conn: GSheetsConnection, vaccine_df: pd.DataFrame, consumables_df: pd.DataFrame) -> None:
    st.write("Edit the tables below, add rows at the bottom if needed, then save each table back to Google Sheets.")
    st.caption("IDs are generated automatically when blank. Use expiry format MMYY, for example 0328.")

    render_editor_section(
        conn,
        vaccine_df,
        section_title="Vaccines",
        worksheet_name=selected_worksheet_name(),
        cleaner=clean_stock_data,
        snapshot_applier=apply_snapshot,
        editor_key="vaccine_stock_editor",
        generic_name_label="Generic name",
        generic_column_name="generic_name",
        save_button_label="Save vaccines to Google Sheet",
        download_file_name="stockvax_vaccines_snapshot.csv",
    )
    st.divider()
    render_editor_section(
        conn,
        consumables_df,
        section_title="Consumables",
        worksheet_name=CONSUMABLES_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        snapshot_applier=apply_consumables_snapshot,
        editor_key="consumables_stock_editor",
        generic_name_label="Generic name / description",
        generic_column_name="generic_name_description",
        save_button_label="Save consumables to Google Sheet",
        download_file_name="stockvax_consumables_snapshot.csv",
    )


def render_app() -> None:
    render_title_banner()

    conn = st.connection("gsheets", type=GSheetsConnection)
    initialize_app_state(conn)
    render_sidebar(conn)
    show_flash_message()
    source_banner()

    raw_stock = clean_stock_data(st.session_state.get(state_key("vaccine", "data")))
    stock_with_metrics = add_inventory_metrics(raw_stock, cleaner=clean_stock_data)
    raw_consumables = clean_consumables_data(st.session_state.get(state_key("consumables", "data")))
    consumables_with_metrics = add_inventory_metrics(raw_consumables, cleaner=clean_consumables_data)

    nurse_tab, consumables_tab, vaccine_overview_tab, consumables_overview_tab, delivery_tab, editor_tab = st.tabs(
        ["Record VACCINE use", "Record CONSUMABLES use", "Vaccine overview", "Consumables overview", "Record delivery", "Stock editor"],
        default="Record VACCINE use",
    )

    with nurse_tab:
        render_nurse_tab(conn, raw_stock)

    with consumables_tab:
        render_consumables_tab(conn, raw_consumables)

    with vaccine_overview_tab:
        render_inventory_overview(
            conn,
            stock_with_metrics,
            section_title="Vaccines",
            tracked_label="Vaccines tracked",
            on_hand_label="Doses on hand",
            item_label_plural="Vaccines",
            value_label="Doses",
            activity_value_label="Vaccines given",
            log_worksheet_name=LOG_WORKSHEET_NAME,
            stock_chart_colors=VACCINE_STOCK_COLORS,
            activity_chart_colors=VACCINE_ACTIVITY_COLORS,
        )

    with consumables_overview_tab:
        render_inventory_overview(
            conn,
            consumables_with_metrics,
            section_title="Consumables",
            tracked_label="Consumables tracked",
            on_hand_label="Items on hand",
            item_label_plural="Consumables",
            value_label="Items",
            activity_value_label="Consumables used",
            log_worksheet_name=CONSUMABLES_LOG_WORKSHEET_NAME,
            stock_chart_colors=CONSUMABLE_STOCK_COLORS,
            activity_chart_colors=CONSUMABLE_ACTIVITY_COLORS,
        )

    with delivery_tab:
        render_delivery_tab(conn, raw_stock, raw_consumables)

    with editor_tab:
        render_editor(conn, raw_stock, raw_consumables)


render_app()
