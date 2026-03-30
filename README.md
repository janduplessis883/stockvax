# StockVax

StockVax is a Streamlit app for managing vaccine stock at Stanhope Mews Surgery with Google Sheets as the live backend.

## What It Does

- Tracks vaccine stock, expected monthly use, order levels, and expiry dates
- Gives a nurse-friendly one-click `Record 1 given` workflow
- Supports delivery intake with quantity entry per vaccine
- Shows overview watchlists for reorders and expiry risk
- Persists changes directly to Google Sheets with `st-gsheets-connection`

## App Tabs

- `Nurse mode`
  Records a single administered vaccine with one click per vaccine card.
- `Record delivery`
  Adds newly delivered stock quantities onto the live sheet.
- `Overview`
  Shows summary metrics, watchlists, and stock usage chart.
- `Stock editor`
  Lets you edit the full stock table and save it back to Google Sheets.

## Data Shape

The app expects these columns in the `vaccines` worksheet:

- `id`
- `brand_name`
- `generic_name`
- `expiry_date`
- `stock_level`
- `expected_monthly_use`
- `order_level`

Expiry dates are stored as `MMYY`, for example `0328`.

## Local Setup

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add Streamlit secrets in `.streamlit/secrets.toml`.
4. Run the app:

```bash
streamlit run streamlit_app.py
```

## Google Sheets Connection

This project uses [`st-gsheets-connection`](https://github.com/streamlit/gsheets-connection) with a Streamlit connection named `gsheets`.

Your `.streamlit/secrets.toml` should include a section like:


Do not commit real secrets to source control.

## Notes

- The app reads and writes the whole worksheet when saving updates.
- Nurse-mode dose recording refreshes from Google Sheets before writing back, so it stays aligned with the latest stock.
- A sample CSV is bundled in `data/stockvax_sample.csv` as a local fallback.
