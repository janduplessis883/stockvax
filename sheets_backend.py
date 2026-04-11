from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import streamlit as st
from st_supabase_connection import SupabaseConnection, execute_query

from app_constants import (
    CONFIG_WORKSHEET_NAME,
    CONSUMABLES_LOG_WORKSHEET_NAME,
    CONSUMABLES_WORKSHEET_NAME,
    DEFAULT_APP_BASE_URL,
    EMERGENCY_DRUG_CATEGORY,
    LOG_WORKSHEET_NAME,
    StockSnapshot,
    VACCINE_CATEGORY,
)
from inventory_logic import clean_consumables_data, clean_stock_data


INVENTORY_SELECT = (
    "id,item_code,category,brand_name,generic_name,expiry_text,"
    "current_stock,expected_monthly_use,order_level,is_active"
)


def inventory_scope_categories(worksheet_name: str | None) -> list[str]:
    if worksheet_name == CONSUMABLES_WORKSHEET_NAME:
        return ["consumable"]
    return [VACCINE_CATEGORY, EMERGENCY_DRUG_CATEGORY]


def movement_scope_categories(log_worksheet_name: str) -> list[str]:
    if log_worksheet_name == CONSUMABLES_LOG_WORKSHEET_NAME:
        return ["consumable"]
    return [VACCINE_CATEGORY, EMERGENCY_DRUG_CATEGORY]


def _execute(query: Any, *, ttl: int | str | None = 0):
    return execute_query(query, ttl=ttl)


def _query_inventory_rows(
    conn: SupabaseConnection,
    *,
    categories: Iterable[str] | None = None,
    include_inactive: bool = False,
    ttl: int | str | None = 0,
) -> list[dict[str, Any]]:
    query = conn.table("inventory_items").select(INVENTORY_SELECT)
    if not include_inactive:
        query = query.eq("is_active", True)

    categories = list(categories or [])
    if len(categories) == 1:
        query = query.eq("category", categories[0])
    elif len(categories) > 1:
        query = query.in_("category", categories)

    response = _execute(query.order("brand_name").order("generic_name"), ttl=ttl)
    return response.data or []


def _inventory_rows_to_app_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "brand_name",
                "generic_name",
                "expiry_date",
                "stock_level",
                "expected_monthly_use",
                "order_level",
                "category",
            ]
        )

    df = pd.DataFrame(
        {
            "id": [row.get("item_code", "") for row in rows],
            "brand_name": [row.get("brand_name", "") for row in rows],
            "generic_name": [row.get("generic_name", "") for row in rows],
            "expiry_date": [row.get("expiry_text", "") for row in rows],
            "stock_level": [row.get("current_stock", 0) for row in rows],
            "expected_monthly_use": [row.get("expected_monthly_use", 0) for row in rows],
            "order_level": [row.get("order_level", 0) for row in rows],
            "category": [row.get("category", "") for row in rows],
        }
    )
    for column in ["id", "brand_name", "generic_name", "expiry_date", "category"]:
        if column in df.columns:
            df[column] = df[column].fillna("")

    return df[
        [
            "id",
            "brand_name",
            "generic_name",
            "expiry_date",
            "stock_level",
            "expected_monthly_use",
            "order_level",
            "category",
        ]
    ].copy()


def _item_payload_from_row(row: pd.Series, *, default_category: str | None = None) -> dict[str, Any]:
    category = str(row.get("category", "")).strip() or (default_category or "")
    return {
        "item_code": str(row.get("id", "")).strip(),
        "category": category,
        "brand_name": str(row.get("brand_name", "")).strip(),
        "generic_name": str(row.get("generic_name", "")).strip(),
        "expiry_text": str(row.get("expiry_date", "")).strip(),
        "current_stock": int(row.get("stock_level", 0)),
        "expected_monthly_use": int(row.get("expected_monthly_use", 0)),
        "order_level": int(row.get("order_level", 0)),
        "is_active": True,
    }


def _current_inventory_item(conn: SupabaseConnection, item_code: str) -> dict[str, Any]:
    response = conn.table("inventory_items").select(INVENTORY_SELECT).eq("item_code", item_code).limit(1).execute()
    rows = response.data or []
    if not rows:
        raise KeyError(item_code)
    return rows[0]


def _insert_movement(
    conn: SupabaseConnection,
    *,
    item_id: str,
    movement_type: str,
    quantity: int,
    stock_before: int,
    stock_after: int,
    source: str,
    notes: str | None = None,
) -> None:
    conn.table("stock_movements").insert(
        {
            "item_id": item_id,
            "movement_type": movement_type,
            "quantity": int(quantity),
            "stock_before": int(stock_before),
            "stock_after": int(stock_after),
            "source": source,
            "notes": notes,
        }
    ).execute()


def apply_inventory_movement(
    conn: SupabaseConnection,
    *,
    item_code: str,
    quantity: int,
    movement_type: str,
    source: str,
    expiry_text: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    item_row = _current_inventory_item(conn, item_code)
    stock_before = int(item_row["current_stock"])

    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    if movement_type == "consume":
        stock_after = stock_before - int(quantity)
        if stock_after < 0:
            raise ValueError("Amount used cannot be more than current stock.")
    elif movement_type == "delivery":
        stock_after = stock_before + int(quantity)
    else:
        raise ValueError(f"Unsupported movement type: {movement_type}")

    update_payload: dict[str, Any] = {"current_stock": stock_after}
    if expiry_text is not None:
        update_payload["expiry_text"] = expiry_text

    previous_expiry = item_row.get("expiry_text") or ""
    conn.table("inventory_items").update(update_payload).eq("id", item_row["id"]).execute()
    try:
        _insert_movement(
            conn,
            item_id=item_row["id"],
            movement_type=movement_type,
            quantity=quantity,
            stock_before=stock_before,
            stock_after=stock_after,
            source=source,
            notes=notes,
        )
    except Exception:
        rollback_payload: dict[str, Any] = {"current_stock": stock_before}
        if expiry_text is not None:
            rollback_payload["expiry_text"] = previous_expiry
        conn.table("inventory_items").update(rollback_payload).eq("id", item_row["id"]).execute()
        raise

    item_row.update(update_payload)
    item_row["current_stock"] = stock_after
    return item_row


def load_inventory_log_data(conn: SupabaseConnection, log_worksheet_name: str) -> pd.DataFrame:
    empty_log = pd.DataFrame(columns=["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"])

    try:
        inventory_rows = _query_inventory_rows(conn, include_inactive=True, ttl=0)
        movement_rows = (
            conn.table("stock_movements")
            .select("item_id,movement_type,quantity,stock_before,occurred_at")
            .eq("movement_type", "consume")
            .order("occurred_at")
            .execute()
            .data
            or []
        )
    except Exception:
        return empty_log

    if not movement_rows or not inventory_rows:
        return empty_log

    inventory_df = pd.DataFrame(inventory_rows)[["id", "item_code", "brand_name", "category"]]
    movement_df = pd.DataFrame(movement_rows)
    merged_df = movement_df.merge(inventory_df, left_on="item_id", right_on="id", how="left")
    scope_categories = movement_scope_categories(log_worksheet_name)
    merged_df = merged_df[merged_df["category"].isin(scope_categories)].copy()
    if merged_df.empty:
        return empty_log

    merged_df["timestamp"] = pd.to_datetime(
        merged_df["occurred_at"],
        errors="coerce",
        utc=True,
        format="ISO8601",
    ).dt.tz_localize(None)
    merged_df = merged_df[merged_df["timestamp"].notna()].copy()
    if merged_df.empty:
        return empty_log

    return (
        merged_df.rename(
            columns={
                "item_code": "vaccine_id",
                "quantity": "doses_given",
            }
        )[["vaccine_id", "brand_name", "stock_before", "doses_given", "timestamp"]]
        .reset_index(drop=True)
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_inventory_log_data_cached(_conn: SupabaseConnection, log_worksheet_name: str) -> pd.DataFrame:
    return load_inventory_log_data(_conn, log_worksheet_name)


def load_stock_movements(conn: SupabaseConnection) -> pd.DataFrame:
    empty_df = pd.DataFrame(
        columns=[
            "brand_name",
            "item_code",
            "category",
            "movement_type",
            "quantity",
            "stock_before",
            "stock_after",
            "timestamp",
            "source",
            "notes",
        ]
    )

    try:
        inventory_rows = _query_inventory_rows(conn, include_inactive=True, ttl=0)
        movement_rows = (
            conn.table("stock_movements")
            .select("item_id,movement_type,quantity,stock_before,stock_after,occurred_at,source,notes")
            .order("occurred_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception:
        return empty_df

    if not movement_rows or not inventory_rows:
        return empty_df

    inventory_df = pd.DataFrame(inventory_rows)[["id", "item_code", "brand_name", "category"]]
    movement_df = pd.DataFrame(movement_rows)
    merged_df = movement_df.merge(inventory_df, left_on="item_id", right_on="id", how="left")
    merged_df["timestamp"] = pd.to_datetime(
        merged_df["occurred_at"],
        errors="coerce",
        utc=True,
        format="ISO8601",
    ).dt.tz_localize(None)
    merged_df["brand_name"] = merged_df["brand_name"].fillna("").astype(str).str.strip()
    merged_df["item_code"] = merged_df["item_code"].fillna("").astype(str).str.strip()
    merged_df["category"] = merged_df["category"].fillna("").astype(str).str.strip()
    merged_df["source"] = merged_df["source"].fillna("").astype(str).str.strip()
    merged_df["notes"] = merged_df["notes"].fillna("").astype(str).str.strip()

    return (
        merged_df[
            [
                "brand_name",
                "item_code",
                "category",
                "movement_type",
                "quantity",
                "stock_before",
                "stock_after",
                "timestamp",
                "source",
                "notes",
            ]
        ]
        .sort_values(["brand_name", "timestamp"], ascending=[True, False], na_position="last")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_stock_movements_cached(_conn: SupabaseConnection) -> pd.DataFrame:
    return load_stock_movements(_conn)


def load_qr_base_url(conn: SupabaseConnection) -> str:
    try:
        rows = (
            conn.table("app_config")
            .select("value")
            .eq("key", "qr_base_url")
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows:
            return DEFAULT_APP_BASE_URL
        value = str(rows[0].get("value") or "").strip().rstrip("/")
        return value or DEFAULT_APP_BASE_URL
    except Exception:
        return DEFAULT_APP_BASE_URL


def save_qr_base_url(conn: SupabaseConnection, base_url: str) -> None:
    conn.table("app_config").upsert({"key": "qr_base_url", "value": base_url}).execute()


def fetch_inventory_snapshot(
    conn: SupabaseConnection,
    worksheet_name: str | None,
    *,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    empty_message: str,
    fallback_loader: Callable[[], pd.DataFrame] | None = None,
) -> StockSnapshot:
    try:
        categories = inventory_scope_categories(worksheet_name)
        rows = _query_inventory_rows(conn, categories=categories, ttl=0)
        df = cleaner(_inventory_rows_to_app_frame(rows))
        if df.empty:
            fallback_data = cleaner(fallback_loader()) if fallback_loader else cleaner(None)
            return StockSnapshot(
                data=fallback_data,
                source_kind="sample" if fallback_loader else "info",
                message=empty_message,
            )
        return StockSnapshot(
            data=df,
            source_kind="supabase",
            message=":material/database_upload: Live data loaded from Supabase.",
        )
    except Exception as exc:
        fallback_data = cleaner(fallback_loader()) if fallback_loader else cleaner(None)
        return StockSnapshot(
            data=fallback_data,
            source_kind="sample" if fallback_loader else "warning",
            message=(
                f"Supabase could not be read ({type(exc).__name__}). Showing the bundled sample dataset instead."
                if fallback_loader
                else f"Supabase could not be read ({type(exc).__name__})."
            ),
        )


def load_live_inventory_frame(
    conn: SupabaseConnection,
    worksheet_name: str | None,
    *,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
) -> pd.DataFrame:
    rows = _query_inventory_rows(conn, categories=inventory_scope_categories(worksheet_name), ttl=0)
    return cleaner(_inventory_rows_to_app_frame(rows))


def fetch_snapshot(conn: SupabaseConnection, worksheet_name: str | None, fallback_loader: Callable[[], pd.DataFrame]) -> StockSnapshot:
    return fetch_inventory_snapshot(
        conn,
        worksheet_name,
        cleaner=clean_stock_data,
        empty_message="The Supabase inventory is empty right now. Sample data has been loaded locally until you save.",
        fallback_loader=fallback_loader,
    )


def fetch_consumables_snapshot(conn: SupabaseConnection) -> StockSnapshot:
    return fetch_inventory_snapshot(
        conn,
        CONSUMABLES_WORKSHEET_NAME,
        cleaner=clean_consumables_data,
        empty_message="The consumables inventory is empty right now.",
        fallback_loader=None,
    )


def sync_inventory_sheet(
    conn: SupabaseConnection,
    worksheet_name: str | None,
    *,
    target_df: pd.DataFrame,
    cleaner: Callable[[pd.DataFrame | None], pd.DataFrame],
    generic_column_name: str,
) -> pd.DataFrame:
    del generic_column_name

    categories = inventory_scope_categories(worksheet_name)
    default_category = "consumable" if categories == ["consumable"] else None
    desired_df = cleaner(target_df)
    current_rows = _query_inventory_rows(conn, categories=categories, include_inactive=True, ttl=0)

    current_by_code = {str(row["item_code"]).strip(): row for row in current_rows}
    current_active_codes = {
        str(row["item_code"]).strip() for row in current_rows if bool(row.get("is_active"))
    }
    desired_codes = {
        str(row["id"]).strip()
        for _, row in desired_df.iterrows()
        if str(row.get("id", "")).strip()
    }

    for _, row in desired_df.iterrows():
        payload = _item_payload_from_row(row, default_category=default_category)
        item_code = payload["item_code"]
        existing = current_by_code.get(item_code)

        if existing is None:
            conn.table("inventory_items").insert(payload).execute()
            continue

        update_payload = {
            "category": payload["category"],
            "brand_name": payload["brand_name"],
            "generic_name": payload["generic_name"],
            "expiry_text": payload["expiry_text"],
            "expected_monthly_use": payload["expected_monthly_use"],
            "order_level": payload["order_level"],
            "is_active": True,
        }
        stock_before = int(existing.get("current_stock") or 0)
        stock_after = payload["current_stock"]
        if stock_before != stock_after:
            update_payload["current_stock"] = stock_after

        conn.table("inventory_items").update(update_payload).eq("id", existing["id"]).execute()
        if stock_before != stock_after:
            _insert_movement(
                conn,
                item_id=existing["id"],
                movement_type="adjustment",
                quantity=abs(stock_after - stock_before),
                stock_before=stock_before,
                stock_after=stock_after,
                source="stock_editor",
                notes="Inventory editor stock change",
            )

    codes_to_deactivate = sorted(current_active_codes - desired_codes)
    for item_code in codes_to_deactivate:
        existing = current_by_code[item_code]
        conn.table("inventory_items").update({"is_active": False}).eq("id", existing["id"]).execute()

    refreshed_rows = _query_inventory_rows(conn, categories=categories, ttl=0)
    return cleaner(_inventory_rows_to_app_frame(refreshed_rows))
