from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "stockvax_sample.csv"
LOGO_PATH = Path(__file__).parent / "data" / "logo.png"
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
LOG_WORKSHEET_NAME = "log"
CONSUMABLES_WORKSHEET_NAME = "consumables"
CONSUMABLES_LOG_WORKSHEET_NAME = "consumables_log"
CONFIG_WORKSHEET_NAME = "config"
APP_TIMEZONE = "Europe/London"
DEFAULT_APP_BASE_URL = "https://stockvax-stanhope.streamlit.app"
VACCINE_CATEGORY_COLUMN = "category"
VACCINE_CATEGORY = "vaccine"
EMERGENCY_DRUG_CATEGORY = "emergency_drug"
VACCINE_STOCK_COLORS = ("#ec5d4b", "#0c1722")
CONSUMABLE_STOCK_COLORS = ("#74a8d1", "#0e1721")
VACCINE_ACTIVITY_COLORS = ("#ec5d4b", "#f3a43b")
CONSUMABLE_ACTIVITY_COLORS = ("#c853a6", "#c853a6")
EXPIRY_MONTH_COLUMN = "expiry_month"
EXPIRY_YEAR_COLUMN = "expiry_year"
EXPIRY_MONTH_OPTIONS = [f"{month:02d}" for month in range(1, 13)]


@dataclass
class StockSnapshot:
    data: object
    source_kind: str
    message: str
