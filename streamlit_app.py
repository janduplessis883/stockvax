from __future__ import annotations

import base64
import html
import re
from typing import Callable
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from st_supabase_connection import SupabaseConnection

from app_constants import (
    BASE_COLUMNS,
    CONSUMABLES_LOG_WORKSHEET_NAME,
    CONSUMABLES_WORKSHEET_NAME,
    DEFAULT_APP_BASE_URL,
    EMERGENCY_DRUG_CATEGORY,
    EXPIRY_MONTH_COLUMN,
    EXPIRY_MONTH_OPTIONS,
    EXPIRY_YEAR_COLUMN,
    LOGO_PATH,
    LOG_WORKSHEET_NAME,
    StockSnapshot,
    VACCINE_ACTIVITY_COLORS,
    VACCINE_CATEGORY,
    VACCINE_CATEGORY_COLUMN,
    VACCINE_STOCK_COLORS,
    CONSUMABLE_ACTIVITY_COLORS,
    CONSUMABLE_STOCK_COLORS,
)
from inventory_logic import (
    add_expiry_editor_columns,
    add_inventory_metrics,
    aggregate_inventory_log_by_month,
    aggregate_vaccine_log_by_day,
    available_expiry_year_options,
    clean_consumables_data,
    clean_stock_data,
    combine_expiry_editor_columns,
    filter_inventory_cards,
    format_expiry_display,
    frames_match,
    load_sample_stock,
    parse_expiry_date,
    resolve_stock_category,
    sort_inventory_cards,
    split_expiry_parts,
)
from sheets_backend import (
    apply_inventory_movement,
    fetch_consumables_snapshot,
    fetch_snapshot,
    load_inventory_log_data,
    load_inventory_log_data_cached,
    load_stock_movements_cached,
    load_live_inventory_frame,
    load_qr_base_url as backend_load_qr_base_url,
    save_qr_base_url as backend_save_qr_base_url,
    sync_inventory_sheet,
)


@st.cache_data(show_spinner=False)
def load_logo_data_uri() -> str:
    encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_logo}"


st.set_page_config(page_title="StockVax", layout="wide")
st.logo(load_logo_data_uri(), size="large")

def selected_worksheet_name() -> str | None:
    worksheet_name = st.session_state.get("worksheet_name", "").strip()
    return worksheet_name or None


def query_param_value(*names: str) -> str:
    for name in names:
        value = st.query_params.get(name)
        if isinstance(value, list):
            value = value[0] if value else ""
        if value:
            return str(value).strip()
    return ""


def is_restock_mode() -> bool:
    return query_param_value("mode").casefold() == "restock"


def is_consume_mode() -> bool:
    return query_param_value("mode").casefold() == "consume"


def inventory_config_for_item_id(
    item_id: str,
) -> tuple[str | None, Callable[[pd.DataFrame | None], pd.DataFrame], str, str]:
    normalized_id = item_id.strip().upper()
    if normalized_id.startswith("CON-"):
        return (
            CONSUMABLES_WORKSHEET_NAME,
            clean_consumables_data,
            "generic_name_description",
            "Consumable",
        )
    return (
        selected_worksheet_name(),
        clean_stock_data,
        "generic_name",
        "Emergency drug" if normalized_id.startswith("EMER-") else "Vaccine",
    )


def fetch_inventory_item(
    conn: SupabaseConnection,
    item_id: str,
) -> tuple[pd.DataFrame, pd.Series, str | None, Callable[[pd.DataFrame | None], pd.DataFrame], str, str]:
    worksheet_name, cleaner, generic_column_name, item_label = inventory_config_for_item_id(item_id)
    latest_df = load_live_inventory_frame(conn, worksheet_name, cleaner=cleaner)
    if latest_df.empty:
        raise ValueError(f"The {item_label.lower()} inventory is empty.")

    match = latest_df["id"].astype(str).str.strip().str.upper() == item_id.strip().upper()
    if not match.any():
        raise KeyError(item_id)

    row_index = latest_df.index[match][0]
    return latest_df, latest_df.loc[row_index], worksheet_name, cleaner, generic_column_name, item_label


def restock_inventory_item(
    conn: SupabaseConnection,
    item_id: str,
    amount: int,
    expiry_value: str | None = None,
) -> tuple[str, int, str]:
    latest_df, row, _, _, _, item_label = fetch_inventory_item(conn, item_id)
    quantity_received = int(amount)
    updated_row = apply_inventory_movement(
        conn,
        item_code=item_id,
        quantity=quantity_received,
        movement_type="delivery",
        source="qr_restock",
        expiry_text=expiry_value,
    )
    return item_label, int(updated_row["current_stock"]), format_expiry_display(updated_row.get("expiry_text"))


def consume_inventory_item(conn: SupabaseConnection, item_id: str, amount: int) -> tuple[str, int]:
    latest_df, row, _, _, _, item_label = fetch_inventory_item(conn, item_id)
    current_stock = int(row["stock_level"])
    if amount <= 0:
        raise ValueError("Amount used must be greater than zero.")
    updated_row = apply_inventory_movement(
        conn,
        item_code=item_id,
        quantity=int(amount),
        movement_type="consume",
        source="qr_consume",
    )
    return item_label, int(updated_row["current_stock"])


def app_url_for_item(item_id: str, mode: str) -> str:
    encoded_item_id = quote(item_id.strip().upper(), safe="")
    return f"{get_qr_base_url()}/?mode={quote(mode, safe='')}&item_id={encoded_item_id}"


def qr_image_url_for_item(item_id: str, mode: str) -> str:
    encoded_url = quote(app_url_for_item(item_id, mode), safe="")
    return f"https://api.qrserver.com/v1/create-qr-code/?size=450x450&margin=20&data={encoded_url}"


def normalize_qr_base_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rstrip("/")


def get_qr_base_url() -> str:
    return normalize_qr_base_url(st.session_state.get("qr_base_url")) or DEFAULT_APP_BASE_URL


def load_qr_base_url(conn: SupabaseConnection) -> str:
    return normalize_qr_base_url(backend_load_qr_base_url(conn)) or DEFAULT_APP_BASE_URL


def save_qr_base_url(conn: SupabaseConnection, base_url: str) -> None:
    backend_save_qr_base_url(conn, base_url)


def queue_qr_base_url_input_sync(base_url: str) -> None:
    normalized_value = normalize_qr_base_url(base_url) or DEFAULT_APP_BASE_URL
    st.session_state["qr_base_url"] = normalized_value
    st.session_state["qr_base_url_input_pending"] = normalized_value


def handle_qr_base_url_change(conn: SupabaseConnection) -> None:
    submitted_value = normalize_qr_base_url(st.session_state.get("qr_base_url_input"))
    if not submitted_value:
        queue_qr_base_url_input_sync(get_qr_base_url())
        set_flash("error", "QR base URL cannot be blank.", icon=":material/error:")
        return
    if not re.match(r"^https?://", submitted_value, flags=re.IGNORECASE):
        queue_qr_base_url_input_sync(get_qr_base_url())
        set_flash("error", "Enter a full URL starting with http:// or https://", icon=":material/error:")
        return
    try:
        save_qr_base_url(conn, submitted_value)
    except Exception as exc:
        queue_qr_base_url_input_sync(get_qr_base_url())
        set_flash("error", f"Could not save QR base URL: {exc}", icon=":material/error:")
        return

    queue_qr_base_url_input_sync(submitted_value)
    set_flash("success", "QR base URL saved to config.", icon=":material/check_circle:")


def render_mobile_inventory_shell_open() -> None:
    st.html(
        """
        <style>
        body {
            margin: 0;
        }
        .restock-shell {
            max-width: 32rem;
            margin: 0 auto;
            padding: 0.85rem 0 1.5rem;
        }
        .restock-card {
            padding: 0 0.15rem 1rem;
        }
        .restock-eyebrow {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.9rem;
            padding: 0.2rem 1.15rem;
            border-radius: 999px;
            background: rgba(219, 102, 83, 0.1);
            border: 1px solid rgba(219, 102, 83, 0.12);
            color: #cf4c34;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .restock-title {
            color: #0c1722;
            font-size: 2rem;
            font-weight: 900;
            line-height: 1.02;
            letter-spacing: -0.04em;
            margin: 0 0 0.22rem;
        }
        .restock-subtitle {
            color: #6b7280;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.2;
            margin: 0 0 0.6rem;
        }
        .restock-expiry-block {
            margin: 0.15rem 0 0.9rem;
        }
        </style>
        <div class="restock-shell">
            <div class="restock-card">
        """
    )


def render_mobile_inventory_shell_close() -> None:
    st.html("</div></div>")


def render_mobile_inventory_header(pill_text: str, item_name: str, item_description: str) -> None:
    st.markdown(
        f"""
        <div class="restock-eyebrow">{html.escape(pill_text)}</div>
        <div class="restock-title">{html.escape(item_name)}</div>
        <div class="restock-subtitle">{html.escape(item_description)}</div>
        """,
        unsafe_allow_html=True,
    )


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

def apply_snapshot(snapshot: StockSnapshot, flash_kind: str | None = None, flash_message: str | None = None) -> None:
    apply_inventory_snapshot("vaccine", snapshot, flash_kind=flash_kind, flash_message=flash_message)


def apply_consumables_snapshot(
    snapshot: StockSnapshot,
    flash_kind: str | None = None,
    flash_message: str | None = None,
) -> None:
    apply_inventory_snapshot("consumables", snapshot, flash_kind=flash_kind, flash_message=flash_message)


def set_flash(kind: str, message: str, icon: str | None = None, color: str | None = None) -> None:
    st.session_state["flash"] = {"kind": kind, "message": message, "icon": icon, "color": color}


def set_toast(message: str, icon: str | None = None) -> None:
    st.session_state.setdefault("toast_queue", []).append({"message": message, "icon": icon})


def initialize_app_state(conn: SupabaseConnection) -> None:
    if "worksheet_name" not in st.session_state:
        st.session_state["worksheet_name"] = ""
    if "qr_base_url" not in st.session_state:
        st.session_state["qr_base_url"] = load_qr_base_url(conn)
    pending_qr_base_url = st.session_state.pop("qr_base_url_input_pending", None)
    if pending_qr_base_url is not None:
        st.session_state["qr_base_url_input"] = pending_qr_base_url
    if "qr_base_url_input" not in st.session_state:
        st.session_state["qr_base_url_input"] = st.session_state["qr_base_url"]

    if state_key("vaccine", "data") not in st.session_state:
        apply_snapshot(fetch_snapshot(conn, selected_worksheet_name(), load_sample_stock))
    if state_key("consumables", "data") not in st.session_state:
        apply_consumables_snapshot(fetch_consumables_snapshot(conn))
    if state_key("vaccine", "log_data") not in st.session_state:
        st.session_state[state_key("vaccine", "log_data")] = load_inventory_log_data_cached(conn, LOG_WORKSHEET_NAME)
    if state_key("consumables", "log_data") not in st.session_state:
        st.session_state[state_key("consumables", "log_data")] = load_inventory_log_data_cached(
            conn, CONSUMABLES_LOG_WORKSHEET_NAME
        )


def show_flash_message() -> None:
    flash = st.session_state.pop("flash", None)
    if not flash:
        return
    if flash["kind"] == "badge":
        st.badge(flash["message"], icon=flash.get("icon"), color=flash.get("color") or "blue")
        return
    getattr(st, flash["kind"])(flash["message"], icon=flash.get("icon"))


def show_sidebar_status_badges() -> None:
    flash = st.session_state.get("flash")
    if flash and flash["kind"] == "badge":
        st.session_state.pop("flash", None)
        st.success(flash["message"], icon=flash.get("icon"))
    source_banner()


def show_toast_message() -> None:
    toasts = st.session_state.pop("toast_queue", [])
    if not toasts:
        return
    for toast in toasts:
        st.toast(toast["message"], icon=toast.get("icon"), duration="long")

def source_banner() -> None:
    sections = [
        ("Vaccines", st.session_state.get(state_key("vaccine", "source_kind")), st.session_state.get(state_key("vaccine", "source_message"), "")),
        ("Consumables", st.session_state.get(state_key("consumables", "source_kind")), st.session_state.get(state_key("consumables", "source_message"), "")),
    ]
    for label, source_kind, message in sections:
        if not message:
            continue
        if source_kind == "supabase":
            badge_label = message.removeprefix(":material/database_upload: ").strip()
            st.success(
                f"{label}: {badge_label}",
                icon=":material/database_upload:",
            )
        elif source_kind == "sample":
            st.warning(f"{label}: {message}")
        elif source_kind == "warning":
            st.error(f"{label}: {message}")
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
            display: flex;
            flex-wrap: wrap;
            align-items: flex-end;
            gap: 0.08em;
            font-family: "Inter", sans-serif;
            font-size: 2.8rem;
            font-weight: 900;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin-bottom: 0.55rem;
            color: {banner_text_color};
        }}
        .hero-title-static {{
            white-space: nowrap;
        }}
        .hero-title-rotator {{
            position: relative;
            display: inline-block;
            min-width: 12.5ch;
            height: 1em;
            overflow: visible;
            transform: translateY(-0.03em);
        }}
        .hero-title-word {{
            position: absolute;
            inset: 0;
            opacity: 0;
            animation: hero-title-fade 9s infinite;
            white-space: nowrap;
        }}
        .hero-title-word:nth-child(1) {{
            animation-delay: 0s;
        }}
        .hero-title-word:nth-child(2) {{
            animation-delay: 3s;
        }}
        .hero-title-word:nth-child(3) {{
            animation-delay: 6s;
        }}
        @keyframes hero-title-fade {{
            0% {{
                opacity: 0;
                transform: translateY(0.18em);
            }}
            8% {{
                opacity: 1;
                transform: translateY(0);
            }}
            25% {{
                opacity: 1;
                transform: translateY(0);
            }}
            33% {{
                opacity: 0;
                transform: translateY(-0.14em);
            }}
            100% {{
                opacity: 0;
                transform: translateY(-0.14em);
            }}
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
            .hero-title-rotator {{
                min-width: 10.8ch;
            }}
            .hero-subtitle {{
                font-size: 0.88rem;
            }}
        }}
        </style>

        <section class="hero-card">
            <div class="hero-kicker">Stanhope Mews Surgery</div>
            <h1 class="hero-title">
                <span class="hero-title-static">Stock</span>
                <span class="hero-title-rotator" aria-label="Vax, EmergencyRx, Consumables">
                    <span class="hero-title-word">Vax</span>
                    <span class="hero-title-word">EmergencyRx</span>
                    <span class="hero-title-word">Consumables</span>
                </span>
            </h1>
            <p class="hero-subtitle">
                Live surgery vaccine, emergency drugs and consumables inventory powered by Python & Streamlit, with clear reorder and expiry visibility.
            </p>
        </section>
        """.format(
        primary_color=primary_color,
        banner_text_color=banner_text_color,
        hero_glow=hex_to_rgba(banner_glow_color, 0.18),
    )
    st.html(banner_html)


def render_sidebar(conn: SupabaseConnection) -> None:
    st.sidebar.header("Supabase connection")
    with st.sidebar:
        st.html(
            """
            <style>
            .sidebar-bars-wrap {
                width: 100%;
                padding: 0.25rem 0 0.9rem;
                background: transparent;
            }
            .sidebar-bars {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 0.2rem;
                height: 5.8rem;
                width: 100%;
                padding: 0 0.5rem;
                box-sizing: border-box;
                background: transparent;
            }
            .sidebar-bars span {
                flex: 1 1 0;
                max-width: 0.95rem;
                height: 2.1rem;
                border-radius: 999px;
                background: linear-gradient(180deg, rgba(233, 136, 117, 0.98) 0%, rgba(219, 102, 83, 0.92) 100%);
                box-shadow: 0 0 0.8rem rgba(219, 102, 83, 0.18);
                transform-origin: center bottom;
                animation: sidebar-bars-rise 3s infinite ease-in-out;
            }
            .sidebar-bars span:nth-child(2n) {
                background: linear-gradient(180deg, rgba(219, 102, 83, 0.95) 0%, rgba(198, 79, 59, 0.88) 100%);
            }
            .sidebar-bars span:nth-child(3n) {
                background: linear-gradient(180deg, rgba(245, 162, 145, 0.98) 0%, rgba(219, 102, 83, 0.9) 100%);
            }
            .sidebar-bars span:nth-child(1) { animation-delay: 0s; }
            .sidebar-bars span:nth-child(2) { animation-delay: 0.16s; }
            .sidebar-bars span:nth-child(3) { animation-delay: 0.32s; }
            .sidebar-bars span:nth-child(4) { animation-delay: 0.48s; }
            .sidebar-bars span:nth-child(5) { animation-delay: 0.64s; }
            .sidebar-bars span:nth-child(6) { animation-delay: 0.8s; }
            .sidebar-bars span:nth-child(7) { animation-delay: 0.96s; }
            .sidebar-bars span:nth-child(8) { animation-delay: 1.12s; }
            .sidebar-bars span:nth-child(9) { animation-delay: 1.28s; }
            .sidebar-bars span:nth-child(10) { animation-delay: 1.44s; }

            @keyframes sidebar-bars-rise {
                0%, 100% {
                    height: 2.1rem;
                    opacity: 0.72;
                    filter: saturate(0.96) brightness(0.98);
                }
                50% {
                    height: 5.15rem;
                    opacity: 1;
                    filter: saturate(1.06) brightness(1.05);
                }
            }
            </style>
            <div class="sidebar-bars-wrap" aria-hidden="true">
                <div class="sidebar-bars">
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            """
        )
        show_sidebar_status_badges()

    st.sidebar.text_input(
        "QR base URL",
        key="qr_base_url_input",
        help="Saved to app_config.qr_base_url and used for all restock and consume QR codes.",
        on_change=handle_qr_base_url_change,
        args=(conn,),
        placeholder=DEFAULT_APP_BASE_URL,
    )

    refresh_clicked = st.sidebar.button("Refresh from Supabase", width="stretch")

    st.sidebar.caption("Editor saves update Supabase inventory records directly, so avoid concurrent edits in multiple browser sessions.")

    if refresh_clicked:
        load_inventory_log_data_cached.clear()
        load_stock_movements_cached.clear()
        apply_snapshot(
            fetch_snapshot(conn, selected_worksheet_name(), load_sample_stock),
        )
        apply_consumables_snapshot(fetch_consumables_snapshot(conn))
        st.session_state[state_key("vaccine", "log_data")] = load_inventory_log_data(conn, LOG_WORKSHEET_NAME)
        st.session_state[state_key("consumables", "log_data")] = load_inventory_log_data(
            conn,
            CONSUMABLES_LOG_WORKSHEET_NAME,
        )
        refreshed_qr_base_url = load_qr_base_url(conn)
        queue_qr_base_url_input_sync(refreshed_qr_base_url)
        set_flash(
            "badge",
            "Data and activity plots refreshed from Supabase.",
            icon=":material/refresh:",
            color="yellow",
        )
        st.rerun()

    with st.sidebar:
        st.html(
            """
            <style>
            .sidebar-signature-wrap {
                margin-top: 1rem;
                padding-bottom: 0.25rem;
                display: flex;
                justify-content: center;
            }
            .sidebar-signature-pill {
                width: 50%;
                border-radius: 999px;
                background: #ffffff;
                color: #51585f;
                text-align: center;
                font-size: 0.78rem;
                font-weight: 900;
                letter-spacing: -0.03em;
                line-height: 1;
                padding: 0.55rem 0.75rem;
                box-shadow: 0 6px 12px rgba(14, 23, 33, 0.12);
                border: 1px solid rgba(14, 23, 33, 0.06);
                transition:
                    background-color 180ms ease,
                    color 180ms ease,
                    transform 180ms ease,
                    box-shadow 180ms ease,
                    border-color 180ms ease;
            }
            .sidebar-signature-pill:hover {
                background: #db6653;
                color: #f4f6f2;
                transform: translateY(8px);
                box-shadow: 0 2px 0 rgba(14, 23, 33, 0.08);
                border-color: 1px solid rgba(14, 23, 33, 0.12);
            }
            .sidebar-signature-pill:active {
                background: #ffffff;
                color: #e49988;
            }
            </style>
            <BR><BR><BR><BR>
            <div class="sidebar-signature-wrap">
                <div class="sidebar-signature-pill">janduplessis883</div>
            </div>
            """
        )


def render_metrics(
    stock_df: pd.DataFrame,
    log_df: pd.DataFrame,
    *,
    tracked_label: str,
    on_hand_label: str,
) -> None:
    total_stock = int(stock_df["stock_level"].sum())
    total_expected_monthly_use = int(stock_df["expected_monthly_use"].sum())
    low_stock_count = int(stock_df["status"].isin(["Reorder", "Out of stock"]).sum())
    expiring_count = int(
        ((stock_df["days_until_expiry"] <= 90) & (stock_df["days_until_expiry"] >= 0) & (stock_df["stock_level"] > 0)).sum()
    )
    months_of_cover = (
        round(total_stock / total_expected_monthly_use, 1)
        if total_expected_monthly_use > 0
        else 0
    )
    stock_delta = total_stock - total_expected_monthly_use if total_expected_monthly_use > 0 else None
    monthly_usage_df = aggregate_inventory_log_by_month(log_df)
    monthly_usage_data = monthly_usage_df["doses_given"].tolist() if not monthly_usage_df.empty else None

    metric_columns = st.columns(4)
    metric_columns[0].metric(tracked_label, len(stock_df))
    metric_columns[1].metric(
        on_hand_label,
        total_stock,
        delta=stock_delta,
        delta_description="vs expected monthly use",
    )
    metric_columns[2].metric("Need action", low_stock_count)
    metric_columns[3].metric("Months of cover", months_of_cover)

    st.badge(f":material/gpp_bad: Expiring in the next 90 days: **{expiring_count}**", color="yellow")


def watchlist_row_styles(row: pd.Series, table_kind: str) -> list[str]:
    text_color = st.get_option("theme.textColor") or "#0c1722"
    low_stock_row_color = "#f9f6f0"
    out_of_stock_row_color = "#faf0ee"
    expiry_row_color = "#edefec"

    background = ""
    border = ""
    status = row.get("status", "")

    if status == "Out of stock":
        background = out_of_stock_row_color
        border = f"4px solid {out_of_stock_row_color}"
    elif status == "Expired stock":
        background = expiry_row_color
        border = f"4px solid {expiry_row_color}"
    elif status == "Reorder":
        background = low_stock_row_color
        border = f"4px solid {low_stock_row_color}"
    elif status == "Expiring soon":
        background = expiry_row_color
        border = f"4px solid {expiry_row_color}"
    elif table_kind == "expiry" and pd.notna(row.get("days_until_expiry")) and int(row.get("days_until_expiry", 9999)) <= 90:
        background = expiry_row_color
        border = f"4px solid {expiry_row_color}"

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
    bar_size: int = 22,
    horizontal: bool = False,
    wrap_in_container: bool = False,
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

    base_chart = alt.Chart(long_plot_df).mark_bar(size=bar_size).encode(
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

    if horizontal:
        chart = base_chart.encode(
            y=alt.Y(
                "brand_name:N",
                sort=plot_df["brand_name"].tolist(),
                axis=alt.Axis(
                    title=None,
                    labelColor=tick_color,
                    labelFontSize=11,
                    labelPadding=8,
                    ticks=False,
                ),
            ),
            x=alt.X(
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
        ).properties(
            title=title,
            height=max(420, len(plot_df) * 22),
        )
    else:
        chart = base_chart.encode(
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
        ).properties(
            title=title,
            height=420,
        )

    chart = chart.configure_title(
        anchor="start",
        color=text_color,
        fontSize=18,
        fontWeight="bold",
    ).configure_view(strokeOpacity=0)

    if wrap_in_container:
        with st.container(border=True):
            st.altair_chart(chart, width="stretch")
        return

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
        line={"color": primary_color, "strokeWidth": 1.5},
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
        strokeWidth=1.5,
    ).encode(
        y="rolling_average:Q",
        tooltip=[
            alt.Tooltip("day:T", title="Day", format="%d %B %Y"),
            alt.Tooltip("rolling_average:Q", title="7-day average", format=".1f"),
        ],
    )

    peak_highlight = (
        alt.Chart(pd.DataFrame([peak_day]))
        .mark_point(shape="diamond", filled=True, size=280, color=accent_color, stroke="white", strokeWidth=1)
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
            height=272,
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


def compact_delivery_label(name: str, quantity: int) -> str:
    cleaned_name = str(name).strip()
    if len(cleaned_name) > 18:
        cleaned_name = f"{cleaned_name[:15].rstrip()}..."
    return f"{cleaned_name} +{int(quantity)}"


def render_delivery_timeline_plot(movement_df: pd.DataFrame) -> None:
    delivery_df = movement_df[
        (movement_df["movement_type"] == "delivery") & movement_df["timestamp"].notna()
    ].copy()
    if delivery_df.empty:
        st.info("No deliveries have been recorded yet.")
        return

    delivery_df["brand_name"] = delivery_df["brand_name"].fillna("").astype(str).str.strip()
    delivery_df["item_code"] = delivery_df["item_code"].fillna("").astype(str).str.strip()
    delivery_df["category"] = delivery_df["category"].fillna("").astype(str).str.strip()
    delivery_df["item_name"] = delivery_df["brand_name"].where(
        delivery_df["brand_name"] != "",
        delivery_df["item_code"],
    )
    delivery_df["quantity"] = pd.to_numeric(delivery_df["quantity"], errors="coerce").fillna(0).astype(int)
    delivery_df = delivery_df[delivery_df["quantity"] > 0].copy()
    if delivery_df.empty:
        st.info("No deliveries with a recorded quantity are available to plot yet.")
        return

    category_palette = {
        VACCINE_CATEGORY: "#ec5d4b",
        EMERGENCY_DRUG_CATEGORY: "#d849ab",
        "consumable": "#74a8d1",
    }
    category_labels = {
        VACCINE_CATEGORY: "Vaccine",
        EMERGENCY_DRUG_CATEGORY: "Emergency drug",
        "consumable": "Consumable",
    }

    delivery_df = delivery_df.sort_values(["timestamp", "quantity", "item_name"], ascending=[True, False, True]).reset_index(drop=True)
    delivery_df["category_label"] = delivery_df["category"].map(category_labels).fillna("Other")
    delivery_df["label"] = [
        compact_delivery_label(item_name, quantity)
        for item_name, quantity in zip(delivery_df["item_name"], delivery_df["quantity"])
    ]
    text_color = st.get_option("theme.textColor") or "#0c1722"
    grid_color = st.get_option("theme.borderColor") or "#d7ded8"
    tick_color = "#7d8386"

    base = alt.Chart(delivery_df).encode(
        x=alt.X(
            "timestamp:T",
            title=None,
            axis=alt.Axis(
                format="%d %b",
                labelAngle=-30,
                labelColor=tick_color,
                labelFontSize=11,
                grid=False,
                domain=False,
                tickColor=grid_color,
            ),
        ),
        y=alt.Y(
            "quantity:Q",
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
            "category_label:N",
            scale=alt.Scale(
                domain=["Vaccine", "Emergency drug", "Consumable"],
                range=[
                    category_palette[VACCINE_CATEGORY],
                    category_palette[EMERGENCY_DRUG_CATEGORY],
                    category_palette["consumable"],
                ],
            ),
            legend=alt.Legend(
                title=None,
                orient="bottom",
                direction="horizontal",
                labelColor=tick_color,
                symbolType="circle",
            ),
        ),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Delivery time", format="%d %B %Y %H:%M"),
            alt.Tooltip("item_name:N", title="Item"),
            alt.Tooltip("category_label:N", title="Category"),
            alt.Tooltip("quantity:Q", title="Delivered volume", format=".0f"),
            alt.Tooltip("item_code:N", title="Item code"),
        ],
    )

    bars = base.mark_bar(
        size=6,
        opacity=0.72,
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
    )
    points = base.mark_circle(
        opacity=0.95,
        size=90,
        stroke="white",
        strokeWidth=1.5,
    )
    labels = base.mark_text(
        align="left",
        baseline="bottom",
        dx=8,
        dy=-4,
        angle=315,
        fontSize=11,
        fontWeight="bold",
        color=text_color,
    ).encode(
        text="label:N",
    )

    chart = (
        alt.layer(bars, points, labels)
        .properties(
            title="Delivery timeline: stock received by date",
            height=400,
        )
        .configure_title(
            anchor="start",
            color=text_color,
            fontSize=18,
            fontWeight="bold",
        )
        .configure_view(strokeOpacity=0)
    )

    total_volume = int(delivery_df["quantity"].sum())
    active_days = int(delivery_df["timestamp"].dt.floor("D").nunique())
    largest_delivery = delivery_df.loc[delivery_df["quantity"].idxmax()]

    st.altair_chart(chart, width="stretch")
    st.caption(
        f"{total_volume} units delivered across {len(delivery_df)} recorded delivery event(s) on {active_days} day(s). "
        f"Largest single delivery was {largest_delivery['item_name']} (+{int(largest_delivery['quantity'])}) "
        f"on {largest_delivery['timestamp']:%d %b %Y}."
    )


def render_split_vaccine_stock_usage_plots(
    stock_df: pd.DataFrame,
    *,
    series_colors: tuple[str, str],
    value_label: str,
) -> None:
    category_series = stock_df.apply(resolve_stock_category, axis=1)
    vaccine_df = stock_df[category_series != EMERGENCY_DRUG_CATEGORY].reset_index(drop=True)
    emergency_df = stock_df[category_series == EMERGENCY_DRUG_CATEGORY].reset_index(drop=True)
    emergency_series_colors = ("#d849ab", series_colors[1])

    render_stock_usage_plot(
        vaccine_df,
        title="Vaccines: current stock vs expected monthly use",
        series_colors=series_colors,
        empty_message="No vaccine data is available to plot yet.",
        item_label="Vaccine",
        value_label=value_label,
        bar_size=33,
    )
    render_stock_usage_plot(
        emergency_df,
        title="Emergency drugs: current stock vs expected monthly use",
        series_colors=emergency_series_colors,
        empty_message="No emergency drug data is available to plot yet.",
        item_label="Emergency drug",
        value_label=value_label,
        bar_size=33,
    )


def split_vaccine_activity_logs(log_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if log_df.empty:
        return log_df.copy(), log_df.copy()

    emergency_mask = log_df["vaccine_id"].fillna("").astype(str).str.upper().str.startswith("EMER-")
    vaccine_log_df = log_df[~emergency_mask].reset_index(drop=True)
    emergency_log_df = log_df[emergency_mask].reset_index(drop=True)
    return vaccine_log_df, emergency_log_df


def render_split_vaccine_activity_plots(
    log_df: pd.DataFrame,
    *,
    activity_colors: tuple[str, str],
) -> None:
    vaccine_log_df, emergency_log_df = split_vaccine_activity_logs(log_df)
    emergency_activity_colors = ("#d849ab", "#8e5db7")

    render_inventory_activity_plot(
        vaccine_log_df,
        title="Daily vaccines activity",
        empty_message="No vaccine activity has been logged yet.",
        item_label_plural="Vaccines",
        activity_value_label="Vaccines given",
        activity_colors=activity_colors,
    )
    render_inventory_activity_plot(
        emergency_log_df,
        title="Daily emergency drugs activity",
        empty_message="No emergency drug activity has been logged yet.",
        item_label_plural="Emergency drugs",
        activity_value_label="Emergency drugs given",
        activity_colors=emergency_activity_colors,
    )


def render_split_vaccine_overview_plots(
    stock_df: pd.DataFrame,
    log_df: pd.DataFrame,
    *,
    stock_chart_colors: tuple[str, str],
    activity_chart_colors: tuple[str, str],
    value_label: str,
) -> None:
    category_series = stock_df.apply(resolve_stock_category, axis=1)
    vaccine_df = stock_df[category_series != EMERGENCY_DRUG_CATEGORY].reset_index(drop=True)
    emergency_df = stock_df[category_series == EMERGENCY_DRUG_CATEGORY].reset_index(drop=True)
    vaccine_log_df, emergency_log_df = split_vaccine_activity_logs(log_df)
    emergency_stock_colors = ("#d849ab", stock_chart_colors[1])
    emergency_activity_colors = ("#d849ab", "#8e5db7")

    render_stock_usage_plot(
        vaccine_df,
        title="Vaccines: current stock vs expected monthly use",
        series_colors=stock_chart_colors,
        empty_message="No vaccine data is available to plot yet.",
        item_label="Vaccine",
        value_label=value_label,
        bar_size=33,
    )
    render_inventory_activity_plot(
        vaccine_log_df,
        title="Daily vaccines activity",
        empty_message="No vaccine activity has been logged yet.",
        item_label_plural="Vaccines",
        activity_value_label="Vaccines given",
        activity_colors=activity_chart_colors,
    )
    render_stock_usage_plot(
        emergency_df,
        title="Emergency drugs: current stock vs expected monthly use",
        series_colors=emergency_stock_colors,
        empty_message="No emergency drug data is available to plot yet.",
        item_label="Emergency drug",
        value_label=value_label,
        bar_size=33,
    )
    render_inventory_activity_plot(
        emergency_log_df,
        title="Daily emergency drugs activity",
        empty_message="No emergency drug activity has been logged yet.",
        item_label_plural="Emergency drugs",
        activity_value_label="Emergency drugs given",
        activity_colors=emergency_activity_colors,
    )


def render_inventory_overview(
    stock_df: pd.DataFrame,
    log_df: pd.DataFrame,
    *,
    section_title: str,
    expired_heading_label: str,
    tracked_label: str,
    on_hand_label: str,
    item_label_plural: str,
    value_label: str,
    activity_value_label: str,
    log_worksheet_name: str,
    stock_chart_colors: tuple[str, str],
    activity_chart_colors: tuple[str, str],
    split_vaccine_categories: bool = False,
) -> None:
    render_metrics(stock_df, log_df, tracked_label=tracked_label, on_hand_label=on_hand_label)

    reorder_df = stock_df[stock_df["status"].isin(["Reorder", "Out of stock"])].copy()
    reorder_df = reorder_df.sort_values(["status", "suggested_order_qty", "brand_name"], ascending=[True, False, True])

    expired_df = stock_df[
        stock_df["days_until_expiry"].notna()
        & (stock_df["days_until_expiry"] < 0)
        & (stock_df["stock_level"] > 0)
    ].copy()
    expired_df = expired_df.sort_values("days_until_expiry")

    expiry_df = stock_df[
        stock_df["days_until_expiry"].notna()
        & (stock_df["days_until_expiry"] >= 0)
        & (stock_df["days_until_expiry"] <= 90)
        & (stock_df["stock_level"] > 0)
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
        st.markdown(f"### Expired {expired_heading_label}")
        if expired_df.empty:
            st.info(f"No expired {item_label_plural.lower()} with stock on hand were found.")
        else:
            st.dataframe(
                styled_watchlist_table(
                    expired_df[
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

        st.subheader("Expiry watchlist")
        if expiry_df.empty:
            st.info(f"No {item_label_plural.lower()} are due to expire in the next 90 days.")
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

    if split_vaccine_categories:
        render_split_vaccine_overview_plots(
            stock_df,
            log_df,
            stock_chart_colors=stock_chart_colors,
            activity_chart_colors=activity_chart_colors,
            value_label=value_label,
        )
    else:
        render_stock_usage_plot(
            stock_df,
            title=f"{section_title}: current stock vs expected monthly use",
            series_colors=stock_chart_colors,
            empty_message=f"No {item_label_plural.lower()} data is available to plot yet.",
            item_label=section_title,
            value_label=value_label,
        )
    if not split_vaccine_categories:
        render_inventory_activity_plot(
            log_df,
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
    conn: SupabaseConnection,
    item_id: str,
    *,
    worksheet_name: str | None,
    log_worksheet_name: str,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    snapshot_applier: Callable[[StockSnapshot], None],
    item_label_singular: str,
    action_word: str,
    toast_icon: str,
) -> None:
    try:
        latest_df = load_live_inventory_frame(conn, worksheet_name, cleaner=cleaner)
    except Exception as exc:
        set_flash(
            "error",
            f"Could not record the {item_label_singular.lower()} because Supabase could not be reached: {exc}",
            icon=":material/error:",
        )
        st.rerun()

    if latest_df.empty:
        set_flash(
            "error",
            f"Could not record the {item_label_singular.lower()} because the inventory is empty.",
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
        updated_row = apply_inventory_movement(
            conn,
            item_code=item_id,
            quantity=1,
            movement_type="consume",
            source="consumables_tab" if item_id.startswith("CON-") else "nurse_tab",
        )
        latest_df.at[row_index, "stock_level"] = int(updated_row["current_stock"])
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
            source_kind="supabase",
            message="Live data loaded from Supabase.",
        ),
    )
    load_inventory_log_data_cached.clear()
    load_stock_movements_cached.clear()
    if log_worksheet_name == CONSUMABLES_LOG_WORKSHEET_NAME:
        st.session_state[state_key("consumables", "log_data")] = load_inventory_log_data(conn, log_worksheet_name)
    else:
        st.session_state[state_key("vaccine", "log_data")] = load_inventory_log_data(conn, log_worksheet_name)
    set_toast(
        f"{item_name} - 1 {item_label_singular.lower()} {action_word}",
        icon=toast_icon,
    )
    set_flash(
        "success",
        f"{item_name}: 1 {item_label_singular.lower()} {action_word} and recorded successfully.",
        icon=":material/check_circle:",
    )
    st.rerun()


def give_vaccine_dose(conn: SupabaseConnection, vaccine_id: str) -> None:
    record_inventory_click(
        conn,
        vaccine_id,
        worksheet_name=selected_worksheet_name(),
        log_worksheet_name=LOG_WORKSHEET_NAME,
        cleaner=clean_stock_data,
        snapshot_applier=apply_snapshot,
        item_label_singular="Emergency drug" if vaccine_id.startswith("EMER-") else "Vaccine",
        action_word="given",
        toast_icon=":material/syringe:",
    )


def use_consumable(conn: SupabaseConnection, consumable_id: str) -> None:
    record_inventory_click(
        conn,
        consumable_id,
        worksheet_name=CONSUMABLES_WORKSHEET_NAME,
        log_worksheet_name=CONSUMABLES_LOG_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        snapshot_applier=apply_consumables_snapshot,
        item_label_singular="Consumable",
        action_word="used",
        toast_icon=":material/clean_hands:",
    )


def render_inventory_action_tab(
    stock_df: pd.DataFrame,
    log_df: pd.DataFrame,
    *,
    conn: SupabaseConnection,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    expander_title: str,
    section_subheader: str,
    chart_item_label: str,
    caption: str,
    empty_message: str,
    button_label: str,
    button_icon: str,
    button_key_prefix: str,
    on_click: Callable[[SupabaseConnection, str], None],
    chart_colors: tuple[str, str],
    chart_bar_size: int = 22,
    brand_font_size: str = "2.0rem",
    generic_font_size: str = "0.95rem",
    expiry_font_size: str = "1.5rem",
    split_vaccine_categories: bool = False,
) -> None:
    with st.expander(expander_title, icon=":material/bar_chart:"):
        stock_with_metrics = add_inventory_metrics(stock_df, cleaner=cleaner)
        if split_vaccine_categories:
            render_split_vaccine_stock_usage_plots(
                stock_with_metrics,
                series_colors=chart_colors,
                value_label="Units",
            )
        else:
            render_stock_usage_plot(
                stock_with_metrics,
                title="Current stock vs expected monthly use",
                series_colors=chart_colors,
                empty_message=empty_message,
                item_label=chart_item_label,
                value_label="Units",
                bar_size=chart_bar_size,
            )
    control_columns = st.columns([1, 1.35])
    with control_columns[0]:
        sort_by_brand_name = st.toggle(
            "Sort by Brand Name",
            value=True,
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
        .nurse-expiry-pill.is-expired {{
            background: rgba(236, 93, 75, 0.12);
            border-color: rgba(236, 93, 75, 0.62);
            color: #ec5d4b;
            box-shadow: 0 4px 12px rgba(236, 93, 75, 0.12);
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

    def render_card_group(group_df: pd.DataFrame, *, title: str | None, empty_group_message: str) -> None:
        if title:
            st.markdown(f"### {title}")
        if group_df.empty:
            st.info(empty_group_message)
            return

        card_columns = st.columns(3)

        for index, (_, row) in enumerate(group_df.iterrows()):
            with card_columns[index % 3]:
                with st.container(border=True):
                    brand_name = row["brand_name"] or row["generic_name"] or "Unnamed item"
                    generic_name = row["generic_name"] or "No description"
                    stock_level = int(row["stock_level"])
                    order_level = int(row["order_level"])
                    expected_monthly_use = int(row.get("expected_monthly_use", 0))
                    stock_delta = stock_level - expected_monthly_use if expected_monthly_use > 0 else None
                    today = pd.Timestamp.today().normalize()
                    expiry_dt = parse_expiry_date(row["expiry_date"])
                    is_expired = pd.notna(expiry_dt) and expiry_dt < today
                    expiry_display = format_expiry_display(row["expiry_date"])
                    expiry_class = "nurse-expiry-pill"
                    if pd.notna(expiry_dt):
                        if is_expired:
                            expiry_class = "nurse-expiry-pill is-expired"
                        elif expiry_dt <= (today + pd.DateOffset(months=1)):
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
                    detail_columns[0].metric(
                        "Stock",
                        f"{stock_level}",
                        delta=stock_delta,
                    )
                    detail_columns[1].metric(":gray[Order level]", f":gray[{order_level}]")

                    if is_expired and stock_level > 0:
                        st.error("Vaccine / drug expired", icon=":material/gpp_bad:", width="stretch")
                    elif stock_level <= 0:
                        st.error("Out of stock", icon=":material/error:", width="stretch")
                    elif stock_level <= order_level and order_level > 0:
                        st.warning("Low stock / Reorder", icon=":material/warning:", width="stretch")
                    else:
                        st.success("Ready to record", icon=":material/check_circle:", width="stretch")

                    if st.button(
                        button_label,
                        key=f"{button_key_prefix}_{row['id']}",
                        type="primary",
                        icon=button_icon,
                        disabled=stock_level <= 0 or is_expired,
                        width="stretch",
                    ):
                        on_click(conn, row["id"])

    if split_vaccine_categories:
        category_series = display_df.apply(resolve_stock_category, axis=1)
        vaccine_display_df = display_df[category_series != EMERGENCY_DRUG_CATEGORY].reset_index(drop=True)
        emergency_display_df = display_df[category_series == EMERGENCY_DRUG_CATEGORY].reset_index(drop=True)
        groups_to_render: list[tuple[pd.DataFrame, str, str]] = [
            (vaccine_display_df, "VACCINES", "No vaccines match the current view."),
            (emergency_display_df, "EMERGENCY DRUGS", "No emergency drugs match the current view."),
        ]
        if len(search_query.strip()) >= 2:
            groups_to_render = [group for group in groups_to_render if not group[0].empty]

        for index, (group_df, title, empty_group_message) in enumerate(groups_to_render):
            if index > 0:
                st.divider()
            render_card_group(group_df, title=title, empty_group_message=empty_group_message)
        return

    render_card_group(display_df, title=None, empty_group_message=empty_message)


def render_nurse_tab(conn: SupabaseConnection, stock_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    render_inventory_action_tab(
        stock_df,
        log_df,
        conn=conn,
        cleaner=clean_stock_data,
        expander_title="Vaccine and emergency drug stock usage bar plot",
        section_subheader="Vaccine & Emergency drug recording",
        chart_item_label="Stock item",
        caption="Keep this tab open during clinic. One click records one vaccine or emergency drug given and updates the live stock level immediately.",
        empty_message="No vaccines are available in the stock list yet.",
        button_label="Record 1 given",
        button_icon=":material/syringe:",
        button_key_prefix="give_dose",
        on_click=give_vaccine_dose,
        chart_colors=VACCINE_STOCK_COLORS,
        split_vaccine_categories=True,
    )


def render_consumables_tab(conn: SupabaseConnection, stock_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    render_inventory_action_tab(
        stock_df,
        log_df,
        conn=conn,
        cleaner=clean_consumables_data,
        expander_title="Consumables stock usage bar plot",
        section_subheader=":material/clean_hands: Consumables recording",
        chart_item_label="Consumable",
        caption="Use this tab to record one consumable used and update the live stock level immediately.",
        empty_message="No consumables are available in the stock list yet.",
        button_label="Record 1 used",
        button_icon=":material/inventory_2:",
        button_key_prefix="use_consumable",
        on_click=use_consumable,
        chart_colors=CONSUMABLE_STOCK_COLORS,
        chart_bar_size=12,
        brand_font_size="1.55rem",
        generic_font_size="0.82rem",
        expiry_font_size="1.2rem",
    )


def apply_inventory_delivery(
    conn: SupabaseConnection,
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

    latest_df = load_live_inventory_frame(conn, worksheet_name, cleaner=cleaner)
    if latest_df.empty:
        raise ValueError(f"The {item_label_plural.lower()} inventory is empty.")

    recorded_items: list[str] = []
    for item_id, amount in positive_deliveries.items():
        match = latest_df["id"] == item_id
        if not match.any():
            continue

        row_index = latest_df.index[match][0]
        item_name = latest_df.at[row_index, "brand_name"] or latest_df.at[row_index, "generic_name"] or item_id
        updated_row = apply_inventory_movement(
            conn,
            item_code=item_id,
            quantity=int(amount),
            movement_type="delivery",
            source="delivery_tab",
        )
        latest_df.at[row_index, "stock_level"] = int(updated_row["current_stock"])
        recorded_items.append(f"{item_name} (+{amount})")

    if not recorded_items:
        raise ValueError(f"None of the {item_label_plural.lower()} delivery rows matched the latest live list.")
    return latest_df, recorded_items


def record_delivery(
    conn: SupabaseConnection,
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
            load_stock_movements_cached.clear()
            apply_snapshot(
                StockSnapshot(
                    data=vaccine_df,
                    source_kind="supabase",
                    message="Live data loaded from Supabase.",
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
            load_stock_movements_cached.clear()
            apply_consumables_snapshot(
                StockSnapshot(
                    data=consumables_df,
                    source_kind="supabase",
                    message="Live data loaded from Supabase.",
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
) -> tuple[str, str, str, str] | None:
    st.markdown(f"## {section_title}")
    if display_df.empty:
        st.info(f"No {section_title.lower()} are available in the stock list yet.")
        return None

    qr_request: tuple[str, str, str, str] | None = None

    for _, row in display_df.iterrows():
        brand_name = row["brand_name"] or row["generic_name"] or "Unnamed item"
        generic_name = row["generic_name"] or "No description"
        item_id = row["id"]
        expiry_display = format_expiry_display(row["expiry_date"]) or "Not set"

        info_column, stock_column, input_column, restock_qr_column, consume_qr_column = st.columns(
            [2.1, 1, 1.05, 0.95, 0.95]
        )
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
        with restock_qr_column:
            restock_qr_clicked = st.form_submit_button(
                "Restock QR",
                key=f"{input_key_prefix}_{item_id}_restock_qr",
                icon=":material/qr_code_2:",
                width="stretch",
            )
            if restock_qr_clicked:
                qr_request = (item_id, brand_name, section_title, "restock")
        with consume_qr_column:
            consume_qr_clicked = st.form_submit_button(
                "Consume QR",
                key=f"{input_key_prefix}_{item_id}_consume_qr",
                icon=":material/qr_code_scanner:",
                width="stretch",
            )
            if consume_qr_clicked:
                qr_request = (item_id, brand_name, section_title, "consume")

    return qr_request


def delivery_quantities_from_state(display_df: pd.DataFrame, *, input_key_prefix: str) -> dict[str, int]:
    return {
        row["id"]: int(st.session_state.get(f"{input_key_prefix}_{row['id']}", 0))
        for _, row in display_df.iterrows()
    }


def render_delivery_form(
    conn: SupabaseConnection,
    display_df: pd.DataFrame,
    *,
    section_title: str,
    input_key_prefix: str,
    submit_label: str,
    vaccine_deliveries: bool,
) -> None:
    if display_df.empty:
        st.markdown(f"## {section_title}")
        st.info(f"No {section_title.lower()} are available in the stock list yet.")
        return

    with st.form(f"{input_key_prefix}_form", clear_on_submit=True, width="stretch", border=False):
        qr_request = render_delivery_section(
            display_df,
            section_title=section_title,
            input_key_prefix=input_key_prefix,
        )
        submitted = st.form_submit_button(
            submit_label,
            type="primary",
            icon=":material/local_shipping:",
            width="stretch",
        )

    if qr_request is not None:
        item_id, item_name, qr_section_title, mode = qr_request
        show_item_qr_dialog(item_id, item_name, qr_section_title, mode)
        return

    if not submitted:
        return

    deliveries = delivery_quantities_from_state(display_df, input_key_prefix=input_key_prefix)
    if vaccine_deliveries:
        record_delivery(conn, deliveries, {})
    else:
        record_delivery(conn, {}, deliveries)

def render_delivery_tab(conn: SupabaseConnection, vaccine_df: pd.DataFrame, consumables_df: pd.DataFrame) -> None:
    st.subheader(":material/local_shipping: Record delivery")
    st.caption(
        "Enter delivered stock and save each section separately so vaccines, emergency drugs, and consumables are recorded independently."
    )
    movement_df = load_stock_movements_cached(conn)
    with st.container(border=True):
        render_delivery_timeline_plot(movement_df)

    vaccine_category_series = vaccine_df.apply(resolve_stock_category, axis=1)
    vaccine_display_df = (
        vaccine_df[vaccine_category_series != EMERGENCY_DRUG_CATEGORY]
        .sort_values(["brand_name", "generic_name"])
        .reset_index(drop=True)
    )
    emergency_display_df = (
        vaccine_df[vaccine_category_series == EMERGENCY_DRUG_CATEGORY]
        .sort_values(["brand_name", "generic_name"])
        .reset_index(drop=True)
    )
    consumables_display_df = consumables_df.sort_values(["brand_name", "generic_name"]).reset_index(drop=True)
    if vaccine_display_df.empty and emergency_display_df.empty and consumables_display_df.empty:
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

    with st.expander("Vaccines", expanded=False, icon=":material/syringe:"):
        render_delivery_form(
            conn,
            vaccine_display_df,
            section_title=":material/syringe: Vaccines",
            input_key_prefix="delivery_vaccine",
            submit_label="Record vaccine delivery",
            vaccine_deliveries=True,
        )
    with st.expander("Emergency drugs", expanded=False, icon=":material/ecg_heart:"):
        render_delivery_form(
            conn,
            emergency_display_df,
            section_title=":material/ecg_heart: Emergency drugs",
            input_key_prefix="delivery_emergency",
            submit_label="Record emergency drug delivery",
            vaccine_deliveries=True,
        )
    with st.expander("Consumables", expanded=False, icon=":material/clean_hands:"):
        render_delivery_form(
            conn,
            consumables_display_df,
            section_title=":material/clean_hands: Consumables",
            input_key_prefix="delivery_consumable",
            submit_label="Record consumable delivery",
            vaccine_deliveries=False,
        )

    st.divider()
    batch_vaccine_column, batch_emergency_column, batch_consumable_column = st.columns(3)
    with batch_vaccine_column:
        vaccine_batch_clicked = st.button(
            ":material/syringe: Vaccines QR sheet",
            type="secondary",
            icon=":material/print:",
            width="stretch",
        )
    with batch_emergency_column:
        emergency_batch_clicked = st.button(
            ":material/ecg_heart: Emergency QR sheet",
            type="secondary",
            icon=":material/print:",
            width="stretch",
        )
    with batch_consumable_column:
        consumable_batch_clicked = st.button(
            ":material/clean_hands: Consumables QR sheet",
            type="secondary",
            icon=":material/print:",
            width="stretch",
        )

    if vaccine_batch_clicked:
        current_sheet = st.session_state.get("delivery_qr_sheet")
        st.session_state["delivery_qr_sheet"] = None if current_sheet == "vaccines" else "vaccines"
    if emergency_batch_clicked:
        current_sheet = st.session_state.get("delivery_qr_sheet")
        st.session_state["delivery_qr_sheet"] = None if current_sheet == "emergency" else "emergency"
    if consumable_batch_clicked:
        current_sheet = st.session_state.get("delivery_qr_sheet")
        st.session_state["delivery_qr_sheet"] = None if current_sheet == "consumables" else "consumables"

    selected_sheet = st.session_state.get("delivery_qr_sheet")
    if selected_sheet == "vaccines":
        st.divider()
        render_batch_qr_sheet(vaccine_display_df, ":material/syringe: Vaccines")
    elif selected_sheet == "emergency":
        st.divider()
        render_batch_qr_sheet(emergency_display_df, ":material/ecg_heart: Emergency drugs")
    elif selected_sheet == "consumables":
        st.divider()
        render_batch_qr_sheet(consumables_display_df, ":material/clean_hands: Consumables")


def render_editor_section(
    conn: SupabaseConnection,
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
    full_stock_df: pd.DataFrame | None = None,
    default_category: str | None = None,
) -> None:
    st.markdown(f"## {section_title}")
    if "is_active" not in stock_df.columns:
        stock_df = stock_df.copy()
        stock_df["is_active"] = True

    inactive_df = stock_df[~stock_df["is_active"].fillna(False).astype(bool)].copy()
    if inactive_df.empty:
        st.caption("Inactive items are only shown in the stock editor.")
    else:
        inactive_labels = [
            f"{row['id']} ({row['brand_name'] or row['generic_name'] or 'Unnamed item'})"
            for _, row in inactive_df.iterrows()
        ]
        st.caption(
            "Inactive items shown here only: "
            + ", ".join(inactive_labels)
            + ". Tick `Is active` to reactivate them."
        )

    raw_editor_columns = BASE_COLUMNS + [column for column in stock_df.columns if column not in BASE_COLUMNS]
    editable_columns: list[str] = []
    for column in raw_editor_columns:
        if column == "expiry_date":
            editable_columns.extend([EXPIRY_MONTH_COLUMN, EXPIRY_YEAR_COLUMN])
        else:
            editable_columns.append(column)

    editable_df = add_expiry_editor_columns(stock_df[raw_editor_columns].copy())
    editable_df = editable_df[editable_columns]
    if default_category and VACCINE_CATEGORY_COLUMN in editable_df.columns:
        editable_df[VACCINE_CATEGORY_COLUMN] = editable_df[VACCINE_CATEGORY_COLUMN].replace("", pd.NA).fillna(default_category)

    expiry_year_options = [""] + available_expiry_year_options(stock_df["expiry_date"].tolist())

    edited_df = st.data_editor(
        editable_df,
        key=editor_key,
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        disabled=["id"],
        column_config={
            "id": st.column_config.TextColumn("Item code"),
            "brand_name": st.column_config.TextColumn("Brand name", required=True),
            "generic_name": st.column_config.TextColumn(generic_name_label, required=True),
            EXPIRY_MONTH_COLUMN: st.column_config.SelectboxColumn(
                "Expiry month",
                options=[""] + EXPIRY_MONTH_OPTIONS,
                required=False,
            ),
            EXPIRY_YEAR_COLUMN: st.column_config.SelectboxColumn(
                "Expiry year",
                options=expiry_year_options,
                required=False,
            ),
            "stock_level": st.column_config.NumberColumn("Stock level", min_value=0, step=1, format="%d"),
            "expected_monthly_use": st.column_config.NumberColumn(
                "Expected monthly use",
                min_value=0,
                step=1,
                format="%d",
            ),
            "order_level": st.column_config.NumberColumn("Order level", min_value=0, step=1, format="%d"),
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=[VACCINE_CATEGORY, EMERGENCY_DRUG_CATEGORY, "consumable"],
                required=False,
            ),
            "is_active": st.column_config.CheckboxColumn("Is active"),
        },
    )

    editor_output_df = edited_df.copy()
    if default_category and VACCINE_CATEGORY_COLUMN in editor_output_df.columns:
        editor_output_df[VACCINE_CATEGORY_COLUMN] = (
            editor_output_df[VACCINE_CATEGORY_COLUMN].replace("", pd.NA).fillna(default_category)
        )

    combined_editor_df, expiry_error = combine_expiry_editor_columns(editor_output_df)
    cleaned_edited_df = cleaner(combined_editor_df)
    changes_pending = not frames_match(cleaned_edited_df, cleaner(stock_df))

    if changes_pending:
        st.info(f"You have unsaved changes in the {section_title.lower()} editor.")
    if expiry_error:
        st.warning(expiry_error)

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
        if expiry_error:
            st.error(expiry_error)
            return
        try:
            data_to_save = cleaned_edited_df
            if full_stock_df is not None:
                untouched_df = full_stock_df[~full_stock_df["id"].isin(stock_df["id"])].copy()
                data_to_save = cleaner(pd.concat([untouched_df, cleaned_edited_df], ignore_index=True))
            saved_df = sync_inventory_sheet(
                conn,
                worksheet_name,
                target_df=data_to_save,
                cleaner=cleaner,
                generic_column_name=generic_column_name,
            )
            load_stock_movements_cached.clear()
            snapshot_applier(
                StockSnapshot(
                    data=saved_df,
                    source_kind="supabase",
                    message="Changes saved to Supabase.",
                ),
                "success",
                f"The {section_title.lower()} table was saved to Supabase.",
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save {section_title.lower()} to Supabase: {exc}")

    if discard_clicked:
        if worksheet_name == CONSUMABLES_WORKSHEET_NAME:
            snapshot = fetch_consumables_snapshot(conn)
            apply_consumables_snapshot(
                snapshot,
                flash_kind="info",
                flash_message=f"Local {section_title.lower()} changes were discarded.",
            )
        else:
            snapshot = fetch_snapshot(conn, worksheet_name, load_sample_stock)
            apply_snapshot(
                snapshot,
                flash_kind="info",
                flash_message=f"Local {section_title.lower()} changes were discarded.",
            )
        st.rerun()


def render_editor(conn: SupabaseConnection, vaccine_df: pd.DataFrame, consumables_df: pd.DataFrame) -> None:
    st.write("Edit the tables below, add rows at the bottom if needed, then save each table back to Supabase.")
    st.caption(
        "IDs are generated automatically when blank. Use expiry format MMYY, for example 0328. "
        "Only the stock editor shows inactive items."
    )

    editor_vaccine_df = vaccine_df.copy()
    editor_consumables_df = consumables_df.copy()
    try:
        editor_vaccine_df = load_live_inventory_frame(
            conn,
            selected_worksheet_name(),
            cleaner=clean_stock_data,
            include_inactive=True,
            include_is_active=True,
        )
        editor_consumables_df = load_live_inventory_frame(
            conn,
            CONSUMABLES_WORKSHEET_NAME,
            cleaner=clean_consumables_data,
            include_inactive=True,
            include_is_active=True,
        )
    except Exception:
        if "is_active" not in editor_vaccine_df.columns:
            editor_vaccine_df["is_active"] = True
        if "is_active" not in editor_consumables_df.columns:
            editor_consumables_df["is_active"] = True

    vaccine_only_df = editor_vaccine_df[
        editor_vaccine_df[VACCINE_CATEGORY_COLUMN].fillna(VACCINE_CATEGORY) == VACCINE_CATEGORY
    ].reset_index(drop=True)
    emergency_drug_df = editor_vaccine_df[
        editor_vaccine_df[VACCINE_CATEGORY_COLUMN].fillna(VACCINE_CATEGORY) == EMERGENCY_DRUG_CATEGORY
    ].reset_index(drop=True)

    render_editor_section(
        conn,
        vaccine_only_df,
        section_title=":material/syringe: Vaccines",
        worksheet_name=selected_worksheet_name(),
        cleaner=clean_stock_data,
        snapshot_applier=apply_snapshot,
        editor_key="vaccine_stock_editor",
        generic_name_label="Generic name",
        generic_column_name="generic_name",
        save_button_label="Save vaccines to Supabase",
        download_file_name="stockvax_vaccines_snapshot.csv",
        full_stock_df=editor_vaccine_df,
        default_category=VACCINE_CATEGORY,
    )
    st.divider()
    render_editor_section(
        conn,
        emergency_drug_df,
        section_title=":material/ecg_heart: Emergency drugs",
        worksheet_name=selected_worksheet_name(),
        cleaner=clean_stock_data,
        snapshot_applier=apply_snapshot,
        editor_key="emergency_stock_editor",
        generic_name_label="Generic name",
        generic_column_name="generic_name",
        save_button_label="Save emergency drugs to Supabase",
        download_file_name="stockvax_emergency_drugs_snapshot.csv",
        full_stock_df=editor_vaccine_df,
        default_category=EMERGENCY_DRUG_CATEGORY,
    )
    st.divider()
    render_editor_section(
        conn,
        editor_consumables_df,
        section_title=":material/clean_hands: Consumables",
        worksheet_name=CONSUMABLES_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        snapshot_applier=apply_consumables_snapshot,
        editor_key="consumables_stock_editor",
        generic_name_label="Generic name / description",
        generic_column_name="generic_name_description",
        save_button_label="Save consumables to Supabase",
        download_file_name="stockvax_consumables_snapshot.csv",
        default_category="consumable",
    )


def render_stock_movements_tab(conn: SupabaseConnection) -> None:
    st.subheader(":material/sync_alt: Stock movements")
    st.caption("Read-only movement history from Supabase, sorted by brand name and newest movement first.")

    movement_df = load_stock_movements_cached(conn)
    if movement_df.empty:
        st.info("No stock movements have been recorded yet.")
        return

    st.dataframe(
        movement_df,
        hide_index=True,
        width="stretch",
        column_config={
            "brand_name": st.column_config.TextColumn("Brand name"),
            "item_code": st.column_config.TextColumn("Item code"),
            "category": st.column_config.TextColumn("Category"),
            "movement_type": st.column_config.TextColumn("Movement type"),
            "quantity": st.column_config.NumberColumn("Quantity", format="%d"),
            "stock_before": st.column_config.NumberColumn("Stock before", format="%d"),
            "stock_after": st.column_config.NumberColumn("Stock after", format="%d"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD/MM/YYYY HH:mm"),
            "source": st.column_config.TextColumn("Source"),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )


def render_restock_mode(conn: SupabaseConnection) -> None:
    item_id = query_param_value("item_id", "item-id").upper()
    render_mobile_inventory_shell_open()

    if not item_id:
        st.error("No item ID was provided in the QR link.")
        render_mobile_inventory_shell_close()
        return

    try:
        latest_df, row, _, _, _, item_label = fetch_inventory_item(conn, item_id)
    except KeyError:
        st.error(f"Item `{item_id}` was not found.")
        render_mobile_inventory_shell_close()
        return
    except Exception as exc:
        st.error(f"Could not load `{item_id}` from Supabase: {exc}")
        render_mobile_inventory_shell_close()
        return

    item_name = row["brand_name"] or row["generic_name"] or item_id
    item_description = row["generic_name"] or item_label
    current_stock = int(row["stock_level"])
    expiry_month_value, expiry_year_value = split_expiry_parts(row["expiry_date"])
    restock_year_options = available_expiry_year_options([row["expiry_date"]])
    default_restock_year = expiry_year_value or f"{pd.Timestamp.today().year % 100:02d}"

    render_mobile_inventory_header("RESTOCK", item_name, item_description)

    st.badge(
        f"Current stock: {current_stock}",
        color="green" if current_stock > 0 else "orange",
        width="stretch",
    )

    with st.form("qr_restock_form", clear_on_submit=False, width="stretch"):
        st.markdown("<div class='restock-expiry-block'>", unsafe_allow_html=True)
        expiry_month_column, expiry_year_column = st.columns(2)
        with expiry_month_column:
            expiry_month = st.selectbox(
                "Expiry Month",
                options=EXPIRY_MONTH_OPTIONS,
                index=EXPIRY_MONTH_OPTIONS.index(expiry_month_value) if expiry_month_value in EXPIRY_MONTH_OPTIONS else 0,
                width="stretch",
            )
        with expiry_year_column:
            expiry_year = st.selectbox(
                "Expiry Year",
                options=restock_year_options,
                index=restock_year_options.index(default_restock_year) if default_restock_year in restock_year_options else 0,
                width="stretch",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        amount = st.number_input("Amount re-stocked", min_value=1, step=1, value=1, width="stretch")
        submitted = st.form_submit_button("Update stock", type="primary", width="stretch")

    if submitted:
        updated_expiry = f"{expiry_month}{expiry_year}"
        try:
            item_label, new_stock, updated_expiry_display = restock_inventory_item(
                conn,
                item_id,
                int(amount),
                expiry_value=updated_expiry,
            )
            expiry_note = f" Expiry updated to {updated_expiry_display}." if updated_expiry else ""
            st.success(f"{item_label} updated successfully. New stock level: {new_stock}.{expiry_note}")
        except Exception as exc:
            st.error(f"Could not update `{item_id}`: {exc}")

    render_mobile_inventory_shell_close()


def render_consume_mode(conn: SupabaseConnection) -> None:
    item_id = query_param_value("item_id", "item-id").upper()
    render_mobile_inventory_shell_open()

    if not item_id:
        st.error("No item ID was provided in the QR link.")
        render_mobile_inventory_shell_close()
        return

    try:
        _, row, _, _, _, item_label = fetch_inventory_item(conn, item_id)
    except KeyError:
        st.error(f"Item `{item_id}` was not found.")
        render_mobile_inventory_shell_close()
        return
    except Exception as exc:
        st.error(f"Could not load `{item_id}` from Supabase: {exc}")
        render_mobile_inventory_shell_close()
        return

    item_name = row["brand_name"] or row["generic_name"] or item_id
    item_description = row["generic_name"] or item_label
    current_stock = int(row["stock_level"])

    render_mobile_inventory_header("CONSUME STOCK", item_name, item_description)

    st.badge(
        f"Current stock: {current_stock}",
        color="green" if current_stock > 0 else "orange",
        width="stretch",
    )

    with st.form("qr_consume_form", clear_on_submit=False, width="stretch"):
        amount = st.number_input("Number Used", min_value=1, step=1, value=1, width="stretch")
        submitted = st.form_submit_button("Update stock", type="primary", width="stretch")

    if submitted:
        try:
            item_label, new_stock = consume_inventory_item(conn, item_id, int(amount))
            st.success(f"{item_label} updated successfully. New stock level: {new_stock}.")
        except Exception as exc:
            st.error(f"Could not update `{item_id}`: {exc}")

    render_mobile_inventory_shell_close()


def render_qr_print_styles() -> None:
    st.html(
        """
        <style>
        .qr-print-block {
            padding: 0.2rem 0.35rem 0.7rem;
        }
        .qr-print-title {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin: 0;
            color: #0c1722;
            font-size: 1.05rem;
            font-weight: 850;
            line-height: 1.08;
            letter-spacing: -0.02em;
        }
        .qr-print-link {
            color: #7d8386;
            font-size: 0.9rem;
            text-decoration: none;
        }
        .qr-print-generic {
            margin: 0.18rem 0 0;
            color: #cf4c34;
            font-size: 0.84rem;
            font-weight: 650;
            line-height: 1.15;
        }
        .qr-print-id {
            margin: 0.15rem 0 0.55rem;
            color: #7d8386;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .qr-print-image {
            width: 9.5rem;
            max-width: 100%;
            display: block;
            margin: 0.05rem 0 0.7rem;
        }
        .qr-print-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.25rem;
            padding: 0.3rem 1.15rem;
            border-radius: 999px;
            background: rgba(219, 102, 83, 0.1);
            border: 1px solid rgba(219, 102, 83, 0.12);
            color: #cf5b3b;
            font-size: 0.92rem;
            font-weight: 850;
            letter-spacing: 0.015em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        </style>
        """
    )


def render_qr_print_block(item_id: str, item_name: str, item_description: str, mode: str) -> None:
    target_url = app_url_for_item(item_id, mode)
    mode_label = "RESTOCK" if mode == "restock" else "CONSUME STOCK"
    st.markdown(
        f"""
        <div class="qr-print-block">
            <div class="qr-print-title">
                <span>{html.escape(item_name)}</span>
                <a class="qr-print-link" href="{html.escape(target_url)}" target="_blank" rel="noopener noreferrer">↪</a>
            </div>
            <div class="qr-print-generic">{html.escape(item_description)}</div>
            <div class="qr-print-id">{html.escape(item_id)}</div>
            <img class="qr-print-image" src="{html.escape(qr_image_url_for_item(item_id, mode))}" alt="{html.escape(mode_label)} QR for {html.escape(item_name)}" />
            <div class="qr-print-pill">{html.escape(mode_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Item QR Code")
def show_item_qr_dialog(item_id: str, item_name: str, section_title: str, mode: str) -> None:
    target_url = app_url_for_item(item_id, mode)
    item_description = "Generic name / description"
    inventory_frames = [
        clean_stock_data(st.session_state.get(state_key("vaccine", "data"))),
        clean_consumables_data(st.session_state.get(state_key("consumables", "data"))),
    ]
    for frame in inventory_frames:
        if frame.empty:
            continue
        match = frame["id"].astype(str).str.strip().str.upper() == item_id.strip().upper()
        if match.any():
            row = frame.loc[match].iloc[0]
            item_description = row.get("generic_name") or row.get("generic_name_description") or item_description
            item_name = row.get("brand_name") or row.get("generic_name") or row.get("generic_name_description") or item_name
            break

    st.badge(f"{section_title}: {item_id}", color="blue")
    render_qr_print_styles()
    render_qr_print_block(item_id, item_name, item_description, mode)
    st.caption(f"Scan this code to open the mobile {('restock' if mode == 'restock' else 'consume stock')} page for this item.")
    st.code(target_url, language=None)


def render_batch_qr_sheet(display_df: pd.DataFrame, section_title: str) -> None:
    st.subheader(f"{section_title} QR sheet")
    st.caption("Print or capture this sheet from the page below. Each item has a restock QR on the left and a consume QR on the right.")
    if display_df.empty:
        st.info(f"No {section_title.lower()} are available for QR printing yet.")
        return

    render_qr_print_styles()

    for _, row in display_df.iterrows():
        item_id = str(row["id"]).strip().upper()
        item_name = row["brand_name"] or row["generic_name"] or item_id
        item_description = row["generic_name"] or "Generic name / description"
        restock_column, consume_column = st.columns(2)
        with restock_column:
            render_qr_print_block(item_id, item_name, item_description, "restock")
        with consume_column:
            render_qr_print_block(item_id, item_name, item_description, "consume")
        st.divider()


def render_app() -> None:
    conn = st.connection("supabase", type=SupabaseConnection)

    if is_restock_mode():
        render_restock_mode(conn)
        return
    if is_consume_mode():
        render_consume_mode(conn)
        return

    render_title_banner()

    initialize_app_state(conn)
    render_sidebar(conn)
    show_flash_message()

    raw_stock = clean_stock_data(st.session_state.get(state_key("vaccine", "data")))
    stock_with_metrics = add_inventory_metrics(raw_stock, cleaner=clean_stock_data)
    raw_consumables = clean_consumables_data(st.session_state.get(state_key("consumables", "data")))
    consumables_with_metrics = add_inventory_metrics(raw_consumables, cleaner=clean_consumables_data)
    vaccine_log_df = st.session_state.get(state_key("vaccine", "log_data"), pd.DataFrame())
    consumables_log_df = st.session_state.get(state_key("consumables", "log_data"), pd.DataFrame())

    navigation_options = [
        ":material/syringe: VACCINE | EMERGENCY Rx",
        ":material/clean_hands: CONSUMABLES",
        ":material/circle_circle: Vaccine | Emergency Rx overview",
        ":material/circle_circle: Consumables overview",
        ":material/package_2: Record delivery",
        ":material/swap_horiz: Stock movements",
        ":material/edit: Stock editor",
    ]
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = navigation_options[0]

    (
        vaccine_tab,
        consumables_tab,
        vaccine_overview_tab,
        consumables_overview_tab,
        delivery_tab,
        movements_tab,
        editor_tab,
    ) = st.tabs(
        navigation_options,
        key="active_section",
        on_change="rerun",
    )

    with vaccine_tab:
        if vaccine_tab.open:
            render_nurse_tab(conn, raw_stock, vaccine_log_df)

    with consumables_tab:
        if consumables_tab.open:
            render_consumables_tab(conn, raw_consumables, consumables_log_df)

    with vaccine_overview_tab:
        if vaccine_overview_tab.open:
            vaccine_overview_df = stock_with_metrics
            live_vaccine_snapshot = fetch_snapshot(conn, selected_worksheet_name(), load_sample_stock)
            if live_vaccine_snapshot.source_kind in {"supabase", "info", "sample"}:
                vaccine_overview_df = add_inventory_metrics(
                    live_vaccine_snapshot.data,
                    cleaner=clean_stock_data,
                )
            render_inventory_overview(
                vaccine_overview_df,
                vaccine_log_df,
                section_title="Vaccines",
                expired_heading_label="Vaccines / Emergency drugs",
                tracked_label="Vaccines tracked",
                on_hand_label="Doses on hand",
                item_label_plural="Vaccines",
                value_label="Doses",
                activity_value_label="Vaccines given",
                log_worksheet_name=LOG_WORKSHEET_NAME,
                stock_chart_colors=VACCINE_STOCK_COLORS,
                activity_chart_colors=VACCINE_ACTIVITY_COLORS,
                split_vaccine_categories=True,
            )

    with consumables_overview_tab:
        if consumables_overview_tab.open:
            consumables_overview_df = consumables_with_metrics
            live_consumables_snapshot = fetch_consumables_snapshot(conn)
            if live_consumables_snapshot.source_kind in {"supabase", "info"}:
                consumables_overview_df = add_inventory_metrics(
                    live_consumables_snapshot.data,
                    cleaner=clean_consumables_data,
                )
            render_inventory_overview(
                consumables_overview_df,
                consumables_log_df,
                section_title="Consumables",
                expired_heading_label="Consumables",
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
        if delivery_tab.open:
            render_delivery_tab(conn, raw_stock, raw_consumables)

    with movements_tab:
        if movements_tab.open:
            render_stock_movements_tab(conn)

    with editor_tab:
        if editor_tab.open:
            render_editor(conn, raw_stock, raw_consumables)

    show_toast_message()


render_app()
