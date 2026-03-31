from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re

import altair as alt
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


def empty_stock_frame() -> pd.DataFrame:
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


def make_unique_ids(values: pd.Series) -> pd.Series:
    generated: list[str] = []
    used: set[str] = set()

    for index, raw_value in enumerate(values, start=1):
        candidate = str(raw_value).strip()
        if candidate.lower() in {"", "nan", "none"}:
            candidate = f"VAX-{index:03d}"

        original = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{original}-{suffix}"
            suffix += 1

        used.add(candidate)
        generated.append(candidate)

    return pd.Series(generated, index=values.index)


def clean_stock_data(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return empty_stock_frame()

    df = raw_df.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(how="all")

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

    df["id"] = make_unique_ids(df["id"])

    ordered_columns = BASE_COLUMNS + [column for column in df.columns if column not in BASE_COLUMNS]
    return df[ordered_columns].reset_index(drop=True)


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


def add_inventory_metrics(stock_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_stock_data(stock_df)
    today = pd.Timestamp.today().normalize()
    df["expiry_dt"] = df["expiry_date"].map(parse_expiry_date)
    df["days_until_expiry"] = (df["expiry_dt"] - today).dt.days

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


def fetch_snapshot(conn: GSheetsConnection, worksheet_name: str | None) -> StockSnapshot:
    try:
        df = clean_stock_data(conn.read(worksheet=worksheet_name, ttl=0))
        if df.empty:
            return StockSnapshot(
                data=clean_stock_data(load_sample_stock()),
                source_kind="sample",
                message="The Google Sheet is empty right now. Sample data has been loaded locally until you save.",
            )
        return StockSnapshot(
            data=df,
            source_kind="google",
            message=":material/database_upload: Live data loaded from Google Sheets.",
        )
    except Exception as exc:
        return StockSnapshot(
            data=clean_stock_data(load_sample_stock()),
            source_kind="sample",
            message=f"Google Sheets could not be read ({type(exc).__name__}). Showing the bundled sample dataset instead.",
        )


def apply_snapshot(snapshot: StockSnapshot, flash_kind: str | None = None, flash_message: str | None = None) -> None:
    st.session_state["stock_data"] = snapshot.data
    st.session_state["source_kind"] = snapshot.source_kind
    st.session_state["source_message"] = snapshot.message
    st.session_state["loaded_at"] = pd.Timestamp.now()
    st.session_state.pop("stock_editor", None)

    if flash_kind and flash_message:
        set_flash(flash_kind, flash_message)


def set_flash(kind: str, message: str, icon: str | None = None) -> None:
    st.session_state["flash"] = {"kind": kind, "message": message, "icon": icon}


def initialize_app_state(conn: GSheetsConnection) -> None:
    if "worksheet_name" not in st.session_state:
        st.session_state["worksheet_name"] = ""

    if "stock_data" not in st.session_state:
        apply_snapshot(fetch_snapshot(conn, selected_worksheet_name()))


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


def source_banner() -> None:
    message = st.session_state.get("source_message", "")
    if st.session_state.get("source_kind") == "google":
        st.success(message)
    else:
        st.warning(message)


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
                Live surgery vaccine inventory powered by Google Sheets & Streamlit, with clear reorder and expiry visibility. Code by janduplessis883.
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
        "Worksheet name",
        key="worksheet_name",
        help="Leave blank to use the first worksheet tab in the spreadsheet from .streamlit/secrets.toml.",
        placeholder="First worksheet tab",
    )

    refresh_clicked = st.sidebar.button("Refresh from Google Sheet", width="stretch")
    sample_clicked = st.sidebar.button("Load sample data locally", width="stretch", disabled=True)

    st.sidebar.caption("Save writes the whole table back to Google Sheets, so avoid concurrent edits in multiple browser sessions.")

    if refresh_clicked:
        apply_snapshot(
            fetch_snapshot(conn, selected_worksheet_name()),
            flash_kind="info",
            flash_message=":material/refresh: Stock data refreshed from Google Sheets.",
        )
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


def render_metrics(stock_df: pd.DataFrame) -> None:
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
    metric_columns[0].metric("Vaccines tracked", len(stock_df))
    metric_columns[1].metric("Doses on hand", total_stock)
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
    return df.style.apply(
        lambda row: watchlist_row_styles(row, table_kind),
        axis=1,
    )


def render_stock_usage_plot(stock_df: pd.DataFrame) -> None:
    plot_df = (
        stock_df[["brand_name", "stock_level", "expected_monthly_use"]]
        .sort_values("brand_name")
        .reset_index(drop=True)
    )
    if plot_df.empty:
        st.info("No stock data is available to plot yet.")
        return

    primary_color = st.get_option("theme.primaryColor") or "#ec5d4b"
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
                    range=[primary_color, text_color],
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
                "series:N",
                sort="ascending",
            ),
            tooltip=[
                alt.Tooltip("brand_name:N", title="Vaccine"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Doses", format=".0f"),
            ],
        )
        .properties(
            title="Current stock vs expected monthly use",
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

    st.altair_chart(chart, use_container_width=True)


def render_overview(stock_df: pd.DataFrame) -> None:
    render_metrics(stock_df)

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
            st.info("No vaccines are currently at or below their order level.")
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
            st.info("No live stock has an expiry date recorded yet.")
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

    render_stock_usage_plot(stock_df)

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


def give_vaccine_dose(conn: GSheetsConnection, vaccine_id: str) -> None:
    try:
        latest_df = clean_stock_data(conn.read(worksheet=selected_worksheet_name(), ttl=0))
    except Exception as exc:
        set_flash(
            "error",
            f"Could not record the vaccination because Google Sheets could not be reached: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    if latest_df.empty:
        set_flash(
            "error",
            "Could not record the vaccination because the vaccine sheet is empty.",
            icon=":material/error:",
        )
        st.rerun()

    match = latest_df["id"] == vaccine_id
    if not match.any():
        set_flash(
            "error",
            "That vaccine could not be found in the latest stock list. Please refresh and try again.",
            icon=":material/error:",
        )
        st.rerun()

    row_index = latest_df.index[match][0]
    vaccine_name = latest_df.at[row_index, "brand_name"] or latest_df.at[row_index, "generic_name"] or vaccine_id
    current_stock = int(latest_df.at[row_index, "stock_level"])

    if current_stock <= 0:
        set_flash(
            "warning",
            f"{vaccine_name} is already at zero stock, so no dose was recorded.",
            icon=":material/warning:",
        )
        st.rerun()

    latest_df.at[row_index, "stock_level"] = current_stock - 1

    try:
        conn.update(worksheet=selected_worksheet_name(), data=latest_df)
    except Exception as exc:
        set_flash(
            "error",
            f"Could not save the updated stock for {vaccine_name}: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    apply_snapshot(
        StockSnapshot(
            data=latest_df,
            source_kind="google",
            message="Live data loaded from Google Sheets.",
        ),
    )
    set_flash(
        "success",
        f"{vaccine_name}: 1 vaccine given and recorded successfully.",
        icon=":material/check_circle:",
    )
    st.rerun()


def render_nurse_tab(conn: GSheetsConnection, stock_df: pd.DataFrame) -> None:
    with st.expander("Stock usage bar plot", icon=":material/bar_chart:"):
        render_stock_usage_plot(add_inventory_metrics(stock_df))
    sort_by_brand_name = st.toggle(
        "Sort by Brand Name",
        value=False,
        help="Off sorts the list by Generic Name.",
    )
    st.subheader("Nurse recording")
    st.caption("Keep this tab open during clinic. One click records one vaccine given and updates the live stock level immediately.")
    st.html(
        """
        <style>
        .nurse-card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.3rem;
        }
        .nurse-card-title-block {
            min-width: 0;
            flex: 1;
        }
        .nurse-brand-name {
            margin: 0 0 0.2rem 0;
            font-size: 1.9rem;
            line-height: 1.05;
            font-weight: 800;
            color: #0c1722;
            letter-spacing: -0.03em;
        }
        .nurse-generic-name {
            margin: 0 0 0.85rem 0;
            font-size: 0.95rem;
            line-height: 1.2;
            font-weight: 500;
            color: #6b7280;
            letter-spacing: -0.01em;
        }
        .nurse-expiry-pill {
            flex-shrink: 0;
            min-width: 6.4rem;
            padding: 0.4rem 0.95rem;
            border-radius: 999px;
            background: rgba(216, 222, 216, 0.32);
            border: 1px solid rgba(216, 222, 216, 0.8);
            box-shadow: 0 4px 12px rgba(12, 23, 34, 0.08);
            font-size: 1.5rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.03em;
            text-align: center;
            color: #d8ded8;
        }
        .nurse-expiry-pill.is-warning {
            background: rgba(211, 159, 90, 0.16);
            border-color: rgba(211, 159, 90, 0.52);
            color: #d39f5a;
            box-shadow: 0 4px 12px rgba(211, 159, 90, 0.16);
        }
        .nurse-expiry-pill.is-urgent {
            background: rgba(236, 93, 75, 0.14);
            border-color: rgba(236, 93, 75, 0.52);
            color: #ec5d4b;
            box-shadow: 0 4px 12px rgba(236, 93, 75, 0.14);
        }
        </style>
        """
    )

    sort_columns = ["brand_name", "generic_name"] if sort_by_brand_name else ["generic_name", "brand_name"]
    display_df = stock_df.sort_values(sort_columns).reset_index(drop=True)
    if display_df.empty:
        st.info("No vaccines are available in the stock list yet.")
        return

    card_columns = st.columns(3)

    for index, (_, row) in enumerate(display_df.iterrows()):
        with card_columns[index % 3]:
            with st.container(border=True):
                brand_name = row["brand_name"] or row["generic_name"] or "Unnamed vaccine"
                generic_name = row["generic_name"] or "No generic name"
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
                    "Record 1 given",
                    key=f"give_dose_{row['id']}",
                    type="primary",
                    icon=":material/syringe:",
                    disabled=stock_level <= 0,
                    width="stretch",
                ):
                    give_vaccine_dose(conn, row["id"])


def record_delivery(conn: GSheetsConnection, deliveries: dict[str, int]) -> None:
    positive_deliveries = {vaccine_id: amount for vaccine_id, amount in deliveries.items() if amount > 0}

    if not positive_deliveries:
        set_flash(
            "info",
            "No delivery quantities were entered.",
            icon=":material/info:",
        )
        st.rerun()

    try:
        latest_df = clean_stock_data(conn.read(worksheet=selected_worksheet_name(), ttl=0))
    except Exception as exc:
        set_flash(
            "error",
            f"Could not record the delivery because Google Sheets could not be reached: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    if latest_df.empty:
        set_flash(
            "error",
            "Could not record the delivery because the vaccine sheet is empty.",
            icon=":material/error:",
        )
        st.rerun()

    recorded_items: list[str] = []

    for vaccine_id, amount in positive_deliveries.items():
        match = latest_df["id"] == vaccine_id
        if not match.any():
            continue

        row_index = latest_df.index[match][0]
        vaccine_name = latest_df.at[row_index, "brand_name"] or latest_df.at[row_index, "generic_name"] or vaccine_id
        latest_df.at[row_index, "stock_level"] = int(latest_df.at[row_index, "stock_level"]) + int(amount)
        recorded_items.append(f"{vaccine_name} (+{amount})")

    if not recorded_items:
        set_flash(
            "error",
            "None of the delivery rows matched the latest vaccine list. Please refresh and try again.",
            icon=":material/error:",
        )
        st.rerun()

    try:
        conn.update(worksheet=selected_worksheet_name(), data=latest_df)
    except Exception as exc:
        set_flash(
            "error",
            f"Could not save the delivery update: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    apply_snapshot(
        StockSnapshot(
            data=latest_df,
            source_kind="google",
            message="Live data loaded from Google Sheets.",
        ),
    )
    summary = ", ".join(recorded_items[:4])
    if len(recorded_items) > 4:
        summary += f" and {len(recorded_items) - 4} more"
    set_flash(
        "success",
        f"Delivery recorded successfully: {summary}.",
        icon=":material/check_circle:",
    )
    st.rerun()


def render_delivery_tab(conn: GSheetsConnection, stock_df: pd.DataFrame) -> None:
    st.subheader("Record delivery")
    st.caption("Enter any new stock delivered today, then save once to add the quantities onto the live stock levels.")

    display_df = stock_df.sort_values(["brand_name", "generic_name"]).reset_index(drop=True)
    if display_df.empty:
        st.info("No vaccines are available in the stock list yet.")
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
        for _, row in display_df.iterrows():
            brand_name = row["brand_name"] or row["generic_name"] or "Unnamed vaccine"
            generic_name = row["generic_name"] or "No generic name"
            vaccine_id = row["id"]
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
                    key=f"delivery_amount_{vaccine_id}",
                    label_visibility="collapsed",
                    placeholder="0",
                    width="stretch",
                )

        submitted = st.form_submit_button(
            "Record delivery",
            type="primary",
            icon=":material/local_shipping:",
            width="stretch",
        )

    if submitted:
        deliveries = {
            row["id"]: int(st.session_state.get(f"delivery_amount_{row['id']}", 0))
            for _, row in display_df.iterrows()
        }
        record_delivery(conn, deliveries)


def render_editor(conn: GSheetsConnection, stock_df: pd.DataFrame) -> None:
    st.write("Edit the table below, add rows at the bottom if needed, then save the whole stock table back to Google Sheets.")
    st.caption("IDs are generated automatically when blank. Use expiry format MMYY, for example 0328.")

    editor_columns = BASE_COLUMNS + [column for column in stock_df.columns if column not in BASE_COLUMNS]
    editable_df = stock_df[editor_columns].copy()

    edited_df = st.data_editor(
        editable_df,
        key="stock_editor",
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        disabled=["id"],
        column_config={
            "id": st.column_config.TextColumn("ID"),
            "brand_name": st.column_config.TextColumn("Brand name", required=True),
            "generic_name": st.column_config.TextColumn("Generic name", required=True),
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

    cleaned_edited_df = clean_stock_data(edited_df)
    changes_pending = not frames_match(cleaned_edited_df, clean_stock_data(stock_df))

    if changes_pending:
        st.info("You have unsaved changes in the editor.")

    button_columns = st.columns([1, 1, 1.2])
    save_clicked = button_columns[0].button("Save to Google Sheet", type="primary", width="stretch")
    discard_clicked = button_columns[1].button("Discard local changes", width="stretch")
    button_columns[2].download_button(
        "Download CSV snapshot",
        data=cleaned_edited_df.to_csv(index=False),
        file_name="stockvax_snapshot.csv",
        mime="text/csv",
        width="stretch",
    )

    if save_clicked:
        try:
            conn.update(worksheet=selected_worksheet_name(), data=cleaned_edited_df)
            apply_snapshot(
                StockSnapshot(
                    data=cleaned_edited_df,
                    source_kind="google",
                    message="Changes saved to Google Sheets.",
                ),
                flash_kind="success",
                flash_message="The stock table was saved to Google Sheets.",
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save to Google Sheets: {exc}")

    if discard_clicked:
        apply_snapshot(
            fetch_snapshot(conn, selected_worksheet_name()),
            flash_kind="info",
            flash_message="Local changes were discarded.",
        )
        st.rerun()


def render_app() -> None:
    render_title_banner()

    conn = st.connection("gsheets", type=GSheetsConnection)
    initialize_app_state(conn)
    render_sidebar(conn)
    show_flash_message()
    source_banner()

    raw_stock = clean_stock_data(st.session_state.get("stock_data"))
    stock_with_metrics = add_inventory_metrics(raw_stock)

    nurse_tab, delivery_tab, overview_tab, editor_tab = st.tabs(
        ["Nurse mode", "Record delivery", "Overview", "Stock editor"],
        default="Nurse mode",
    )

    with nurse_tab:
        render_nurse_tab(conn, raw_stock)

    with delivery_tab:
        render_delivery_tab(conn, raw_stock)

    with overview_tab:
        render_overview(stock_with_metrics)

    with editor_tab:
        render_editor(conn, raw_stock)


render_app()
