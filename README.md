# Rossmann Store Network Analysis

End-to-end retail analytics project on the Rossmann Store Sales dataset (1,115 stores, Germany, 2013–2015). Built as a capstone project for the WBS Coding School Data Analytics & AI course.

## What's inside
- `notebooks/01_data_cleaning.ipynb` — data cleaning, missing values, feature engineering
- `notebooks/02_eda_network_level.ipynb` — network-wide trends and seasonality
- `notebooks/03_store_segmentation.ipynb` — store type / assortment / competition analysis
- `notebooks/04_promo_ab_test.ipynb` — formal A/B test on promotions (Mann-Whitney, Cohen's d)
- `notebooks/05_holidays_seasonality.ipynb` — holiday and school-break effects
- `notebooks/06_store_ranking.ipynb` — store-level ranking and growth analysis
- `dashboard.html` — self-contained interactive dashboard (open directly in a browser, no server needed)
- `dashR.twbx` — Tableau Public packaged workbook version of the dashboard
- `notebooks/store_summary.csv` — aggregated per-store metrics (used by the dashboard)
- `data/tableau_csvs/` — pre-aggregated CSVs used to build the Tableau version

## Data
Raw data is not included in this repo (files are too large for GitHub). Download `train.csv`, `store.csv`, `test.csv` from the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales/data) and place them in a local `data/` folder before running the notebooks.

## Key findings
- Promotions increase sales by ~39% on average (medium-to-large effect size, Cohen's d ≈ 0.80)
- Store type "b" shows the highest sales, but represents only 17 of 1,115 stores — a small-sample result, not a network-wide strategy signal
- Total sales and year-over-year growth are nearly uncorrelated across stores
- Network-wide sales dropped 5.3% in 2014 vs 2013, then recovered 6.6% into 2015

## How to run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter`
2. Download the data (see above) into a `data/` folder
3. Run notebooks in order, 01 through 06
