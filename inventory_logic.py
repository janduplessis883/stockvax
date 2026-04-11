from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from app_constants import (
    BASE_COLUMNS,
    CONSUMABLES_WORKSHEET_NAME,
    EMERGENCY_DRUG_CATEGORY,
    EXPIRY_MONTH_COLUMN,
    EXPIRY_MONTH_OPTIONS,
    EXPIRY_YEAR_COLUMN,
    NUMBER_COLUMNS,
    SAMPLE_DATA_PATH,
    TEXT_COLUMNS,
    VACCINE_CATEGORY,
    VACCINE_CATEGORY_COLUMN,
)


EXPIRY_PATTERN = re.compile(r"^\d{4}$")


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


def normalize_stock_category(value: object) -> str:
    normalized = normalize_column_name(value)
    return EMERGENCY_DRUG_CATEGORY if normalized == EMERGENCY_DRUG_CATEGORY else VACCINE_CATEGORY


def resolve_stock_category(row: pd.Series) -> str:
    candidate = normalize_stock_category(row.get(VACCINE_CATEGORY_COLUMN, VACCINE_CATEGORY))
    item_id = str(row.get("id", "")).strip().upper()
    if item_id.startswith("EMER-"):
        return EMERGENCY_DRUG_CATEGORY
    return candidate


def make_unique_stock_ids(df: pd.DataFrame) -> pd.Series:
    generated: list[str] = []
    used: set[str] = set()
    next_numbers = {
        VACCINE_CATEGORY: 1,
        EMERGENCY_DRUG_CATEGORY: 1,
    }

    for _, row in df.iterrows():
        candidate = str(row.get("id", "")).strip()
        category = normalize_stock_category(row.get(VACCINE_CATEGORY_COLUMN, VACCINE_CATEGORY))

        if candidate.lower() in {"", "nan", "none"}:
            prefix = "EMER" if category == EMERGENCY_DRUG_CATEGORY else "VAX"
            candidate = f"{prefix}-{next_numbers[category]:03d}"
            while candidate in used:
                next_numbers[category] += 1
                candidate = f"{prefix}-{next_numbers[category]:03d}"
            next_numbers[category] += 1
        else:
            original = candidate
            suffix = 2
            while candidate in used:
                candidate = f"{original}-{suffix}"
                suffix += 1

        used.add(candidate)
        generated.append(candidate)

    return pd.Series(generated, index=df.index)


def clean_inventory_data(
    raw_df: pd.DataFrame | None,
    *,
    blank_id_template: str | None = None,
    extra_text_columns: tuple[str, ...] = (),
    id_generator: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
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

    for column in extra_text_columns:
        if column not in df.columns:
            df[column] = ""

    for column in (*TEXT_COLUMNS, *extra_text_columns):
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["expiry_date"] = df["expiry_date"].map(normalize_expiry_text)

    for column in NUMBER_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).round().astype(int)

    if id_generator is not None:
        df["id"] = id_generator(df)
    elif blank_id_template is not None:
        df["id"] = make_unique_ids(df["id"], blank_id_template=blank_id_template)

    ordered_columns = BASE_COLUMNS + [column for column in df.columns if column not in BASE_COLUMNS]
    return df[ordered_columns].reset_index(drop=True)


def clean_stock_data(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    df = clean_inventory_data(
        raw_df,
        extra_text_columns=(VACCINE_CATEGORY_COLUMN,),
        id_generator=make_unique_stock_ids,
    )
    if VACCINE_CATEGORY_COLUMN not in df.columns:
        df[VACCINE_CATEGORY_COLUMN] = VACCINE_CATEGORY
    df[VACCINE_CATEGORY_COLUMN] = df.apply(resolve_stock_category, axis=1)
    return df


def clean_consumables_data(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    df = clean_inventory_data(
        raw_df,
        blank_id_template="CON-{index:03d}",
        extra_text_columns=(VACCINE_CATEGORY_COLUMN,),
    )
    df[VACCINE_CATEGORY_COLUMN] = df[VACCINE_CATEGORY_COLUMN].replace("", pd.NA).fillna("consumable")
    return df


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
    if isinstance(value, pd.Timestamp):
        return value.normalize()

    expiry = normalize_expiry_text(value)
    if EXPIRY_PATTERN.fullmatch(expiry):
        month = int(expiry[:2])
        year = 2000 + int(expiry[2:])
        return pd.Period(f"{year}-{month:02d}", freq="M").end_time.normalize()

    parsed_date = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed_date):
        return pd.NaT
    return pd.Timestamp(parsed_date).normalize()


def format_expiry_display(value: object) -> str:
    expiry = normalize_expiry_text(value)
    if not EXPIRY_PATTERN.fullmatch(expiry):
        return ""
    return f"{expiry[:2]}/{expiry[2:]}"


def split_expiry_parts(value: object) -> tuple[str, str]:
    expiry = normalize_expiry_text(value)
    if not EXPIRY_PATTERN.fullmatch(expiry):
        return "", ""
    return expiry[:2], expiry[2:]


def available_expiry_year_options(values: pd.Series | list[object] | None = None) -> list[str]:
    current_year = pd.Timestamp.today().year
    year_options = {f"{year % 100:02d}" for year in range(current_year - 2, current_year + 16)}

    if values is not None:
        for value in values:
            _, year = split_expiry_parts(value)
            if year:
                year_options.add(year)

    return sorted(year_options, key=int)


def add_expiry_editor_columns(df: pd.DataFrame) -> pd.DataFrame:
    editor_df = df.copy()
    expiry_parts = editor_df["expiry_date"].map(split_expiry_parts)
    editor_df[EXPIRY_MONTH_COLUMN] = expiry_parts.map(lambda parts: parts[0])
    editor_df[EXPIRY_YEAR_COLUMN] = expiry_parts.map(lambda parts: parts[1])
    return editor_df.drop(columns=["expiry_date"], errors="ignore")


def combine_expiry_editor_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    editor_df = df.copy()
    month_series = editor_df.get(EXPIRY_MONTH_COLUMN, "").fillna("").astype(str).str.strip()
    year_series = editor_df.get(EXPIRY_YEAR_COLUMN, "").fillna("").astype(str).str.strip()

    partial_mask = (month_series == "") ^ (year_series == "")
    if partial_mask.any():
        row_number = int(partial_mask[partial_mask].index[0]) + 1
        return editor_df, f"Expiry month and year must both be selected, or both left blank. Check row {row_number}."

    editor_df["expiry_date"] = [f"{month}{year}" if month and year else "" for month, year in zip(month_series, year_series)]
    editor_df = editor_df.drop(columns=[EXPIRY_MONTH_COLUMN, EXPIRY_YEAR_COLUMN], errors="ignore")
    return editor_df, None


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
    expiry_series = pd.Series([parse_expiry_date(value) for value in df["expiry_date"].tolist()], index=df.index)
    df["expiry_dt"] = pd.to_datetime(expiry_series, errors="coerce")
    df["days_until_expiry"] = ((df["expiry_dt"] - today).dt.days).astype("Int64")

    expected_use = df["expected_monthly_use"].astype(float)
    stock_level = df["stock_level"].astype(float)
    df["months_of_cover"] = np.where(expected_use > 0, np.round(stock_level / expected_use, 1), np.nan)
    df["suggested_order_qty"] = (df["expected_monthly_use"] - df["stock_level"]).clip(lower=0).astype(int)
    df["status"] = df.apply(status_for_row, axis=1)
    return df


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
        header_candidates[index] in allowed_values for index, allowed_values in enumerate(expected_columns)
    )


def parse_log_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return pd.NaT

    common_formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    )
    for timestamp_format in common_formats:
        parsed = pd.to_datetime(text, format=timestamp_format, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed)

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed)


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


def aggregate_inventory_log_by_month(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame(columns=["month", "doses_given"])

    monthly_item_df = (
        log_df.assign(
            item_id=log_df["vaccine_id"].replace("", pd.NA).fillna(log_df["brand_name"]),
            month=log_df["timestamp"].dt.to_period("M").dt.to_timestamp(),
        )
        .groupby(["month", "item_id"], as_index=False)["doses_given"]
        .sum()
    )
    return monthly_item_df.groupby("month", as_index=False)["doses_given"].sum().sort_values("month").reset_index(drop=True)


def monthly_usage_series_for_item(log_df: pd.DataFrame, item_id: str) -> list[float] | None:
    if log_df.empty:
        return None

    item_log_df = log_df[log_df["vaccine_id"] == item_id].copy()
    if item_log_df.empty:
        return None

    item_monthly_df = (
        item_log_df.assign(month=item_log_df["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["doses_given"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
    return item_monthly_df["doses_given"].astype(float).tolist()


def frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.fillna("").reset_index(drop=True).equals(right.fillna("").reset_index(drop=True))


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

    sorted_df = sort_df.sort_values([primary_sort, secondary_sort, "_id_sort"], kind="mergesort").reset_index(drop=True)
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
    if VACCINE_CATEGORY_COLUMN in stock_df.columns:
        match_mask = match_mask | stock_df[VACCINE_CATEGORY_COLUMN].map(normalized_sort_text).str.contains(
            normalized_query,
            regex=False,
        )
    return stock_df[match_mask].reset_index(drop=True)
