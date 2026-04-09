from __future__ import annotations

from collections.abc import Callable

from gspread import WorksheetNotFound
from gspread.cell import Cell
from gspread.worksheet import Worksheet
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

from app_constants import (
    APP_TIMEZONE,
    CONFIG_WORKSHEET_NAME,
    CONSUMABLES_LOG_WORKSHEET_NAME,
    CONSUMABLES_WORKSHEET_NAME,
    DEFAULT_APP_BASE_URL,
    LOG_WORKSHEET_NAME,
    StockSnapshot,
)
from inventory_logic import (
    clean_consumables_data,
    clean_stock_data,
    first_log_row_is_header,
    normalize_column_name,
    parse_log_timestamp,
    prepare_inventory_sheet_data,
)


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


def _sheet_scalar(value: object) -> object:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _comparison_scalar(value: object) -> object:
    value = _sheet_scalar(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _worksheet_headers(worksheet: Worksheet) -> list[str]:
    return worksheet.row_values(1)


def _ensure_worksheet_headers(worksheet: Worksheet, desired_headers: list[str]) -> list[str]:
    raw_headers = _worksheet_headers(worksheet)
    if not raw_headers:
        header_cells = [Cell(1, index, header) for index, header in enumerate(desired_headers, start=1)]
        if header_cells:
            worksheet.update_cells(header_cells, value_input_option="USER_ENTERED")
        return desired_headers.copy()

    normalized_headers = [normalize_column_name(header) for header in raw_headers]
    new_header_cells: list[Cell] = []
    for header in desired_headers:
        normalized_header = normalize_column_name(header)
        if normalized_header not in normalized_headers:
            raw_headers.append(header)
            normalized_headers.append(normalized_header)
            new_header_cells.append(Cell(1, len(raw_headers), header))

    if new_header_cells:
        worksheet.update_cells(new_header_cells, value_input_option="USER_ENTERED")
    return raw_headers


def _header_index_map(headers: list[str]) -> dict[str, int]:
    return {
        normalize_column_name(header): index
        for index, header in enumerate(headers, start=1)
        if str(header).strip()
    }


def _worksheet_row_numbers_by_id(worksheet: Worksheet, id_column_index: int) -> dict[str, int]:
    return {
        str(value).strip(): row_number
        for row_number, value in enumerate(worksheet.col_values(id_column_index)[1:], start=2)
        if str(value).strip()
    }


def update_inventory_cells_by_id(
    worksheet: Worksheet,
    updates_by_id: dict[str, dict[str, object]],
) -> None:
    if not updates_by_id:
        return

    headers = _ensure_worksheet_headers(worksheet, ["id"])
    header_index = _header_index_map(headers)
    if "id" not in header_index:
        raise KeyError("id")

    row_numbers_by_id = _worksheet_row_numbers_by_id(worksheet, header_index["id"])
    cells: list[Cell] = []
    missing_ids: list[str] = []

    for item_id, updates in updates_by_id.items():
        row_number = row_numbers_by_id.get(str(item_id).strip())
        if row_number is None:
            missing_ids.append(str(item_id))
            continue

        for column_name, value in updates.items():
            normalized_column = normalize_column_name(column_name)
            column_index = header_index.get(normalized_column)
            if column_index is None:
                raise KeyError(column_name)
            cells.append(Cell(row_number, column_index, _sheet_scalar(value)))

    if missing_ids:
        raise KeyError(", ".join(missing_ids))
    if cells:
        worksheet.update_cells(cells, value_input_option="USER_ENTERED")


def _row_values_for_headers(row: pd.Series, headers: list[str]) -> list[object]:
    normalized_values = {normalize_column_name(column): _sheet_scalar(value) for column, value in row.items()}
    return [normalized_values.get(normalize_column_name(header), "") for header in headers]


def sync_inventory_sheet(
    conn: GSheetsConnection,
    worksheet_name: str | None,
    *,
    target_df: pd.DataFrame,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    generic_column_name: str,
) -> pd.DataFrame:
    worksheet = select_live_worksheet(conn, worksheet_name)
    desired_df = prepare_inventory_sheet_data(cleaner(target_df), generic_column_name=generic_column_name)
    live_df = cleaner(conn.read(worksheet=worksheet_name, ttl=0))
    live_export_df = prepare_inventory_sheet_data(live_df, generic_column_name=generic_column_name)

    headers = _ensure_worksheet_headers(worksheet, desired_df.columns.tolist())
    header_index = _header_index_map(headers)
    if "id" not in header_index:
        raise KeyError("id")

    row_numbers_by_id = _worksheet_row_numbers_by_id(worksheet, header_index["id"])
    live_by_id = {str(row["id"]).strip(): row for _, row in live_export_df.iterrows()}
    target_by_id = {str(row["id"]).strip(): row for _, row in desired_df.iterrows()}

    changed_cells: list[Cell] = []
    for item_id, target_row in target_by_id.items():
        if item_id not in live_by_id:
            continue
        row_number = row_numbers_by_id.get(item_id)
        if row_number is None:
            raise KeyError(item_id)

        live_row = live_by_id[item_id]
        for column_name in desired_df.columns:
            if column_name == "id":
                continue
            if _comparison_scalar(live_row.get(column_name)) == _comparison_scalar(target_row.get(column_name)):
                continue
            column_index = header_index.get(normalize_column_name(column_name))
            if column_index is None:
                raise KeyError(column_name)
            changed_cells.append(Cell(row_number, column_index, _sheet_scalar(target_row.get(column_name))))

    if changed_cells:
        worksheet.update_cells(changed_cells, value_input_option="USER_ENTERED")

    deleted_row_numbers = [
        row_numbers_by_id[item_id]
        for item_id in live_by_id
        if item_id not in target_by_id and item_id in row_numbers_by_id
    ]
    for row_number in sorted(deleted_row_numbers, reverse=True):
        worksheet.delete_rows(row_number)

    for item_id, target_row in target_by_id.items():
        if item_id in live_by_id:
            continue
        worksheet.append_row(_row_values_for_headers(target_row, headers), value_input_option="USER_ENTERED")

    return cleaner(target_df)


def append_vaccine_log_row(
    log_worksheet: Worksheet,
    *,
    vaccine_id: str,
    vaccine_name: str,
    stock_left_before_click: int,
    doses_given: int = 1,
) -> None:
    timestamp = pd.Timestamp.now(tz=APP_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    log_worksheet.append_row(
        [vaccine_id, vaccine_name, stock_left_before_click, doses_given, timestamp],
        value_input_option="USER_ENTERED",
    )


def load_inventory_log_data(conn: GSheetsConnection, log_worksheet_name: str) -> pd.DataFrame:
    empty_log = pd.DataFrame(columns=["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"])

    try:
        log_df = conn.read(worksheet=log_worksheet_name, ttl=0, header=None)
    except WorksheetNotFound:
        return empty_log
    except Exception:
        return empty_log

    if log_df is None or log_df.empty:
        return empty_log

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
    trimmed["timestamp"] = trimmed["timestamp"].map(parse_log_timestamp)
    trimmed = trimmed[trimmed["timestamp"].notna()].copy()

    return trimmed.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_inventory_log_data_cached(_conn: GSheetsConnection, log_worksheet_name: str) -> pd.DataFrame:
    return load_inventory_log_data(_conn, log_worksheet_name)


def load_qr_base_url(conn: GSheetsConnection) -> str:
    try:
        config_worksheet = select_live_worksheet(conn, CONFIG_WORKSHEET_NAME)
        value = str(config_worksheet.acell("A1").value or "").strip().rstrip("/")
        return value or DEFAULT_APP_BASE_URL
    except WorksheetNotFound:
        return DEFAULT_APP_BASE_URL
    except Exception:
        return DEFAULT_APP_BASE_URL


def save_qr_base_url(conn: GSheetsConnection, base_url: str) -> None:
    config_worksheet = select_live_worksheet(conn, CONFIG_WORKSHEET_NAME)
    config_worksheet.update_acell("A1", base_url)


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


def fetch_snapshot(conn: GSheetsConnection, worksheet_name: str | None, fallback_loader: Callable[[], pd.DataFrame]) -> StockSnapshot:
    return fetch_inventory_snapshot(
        conn,
        worksheet_name,
        cleaner=clean_stock_data,
        empty_message="The Google Sheet is empty right now. Sample data has been loaded locally until you save.",
        fallback_loader=fallback_loader,
    )


def fetch_consumables_snapshot(conn: GSheetsConnection) -> StockSnapshot:
    return fetch_inventory_snapshot(
        conn,
        CONSUMABLES_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        empty_message="The consumables sheet is empty right now.",
        fallback_loader=None,
    )
