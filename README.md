# StockVax

StockVax is a Streamlit app for managing vaccine stock at Stanhope Mews Surgery with Supabase as the live backend.

## What It Does

- Tracks vaccine stock, expected monthly use, order levels, and expiry dates
- Gives a nurse-friendly one-click `Record 1 given` workflow
- Supports delivery intake with quantity entry per vaccine
- Shows overview watchlists for reorders and expiry risk
- Persists changes directly to Supabase with `st-supabase-connection`

## App Tabs

- `Nurse mode`
  Records a single administered vaccine with one click per vaccine card.
- `Record delivery`
  Adds newly delivered stock quantities onto the live inventory.
- `Overview`
  Shows summary metrics, watchlists, and stock usage chart.
- `Stock editor`
  Lets you edit the full stock table and save it back to Supabase.

## Data Shape

The app expects these inventory columns:

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

## Supabase Connection

This project uses [`st-supabase-connection`](https://pypi.org/project/st-supabase-connection/) with a Streamlit connection named `supabase`.

Your `.streamlit/secrets.toml` should include a section like:

```toml
[connections.supabase]
SUPABASE_URL = "xxxx"
SUPABASE_KEY = "xxxx"
```


Do not commit real secrets to source control.

## Notes

- The app reads and writes inventory through Supabase tables.
- Nurse-mode dose recording updates the matching stock row and appends a `consume` movement to `stock_movements`.
- A sample CSV is bundled in `data/stockvax_sample.csv` as a local fallback.
