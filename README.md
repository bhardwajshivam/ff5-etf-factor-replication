# ETF Factor Proxy Replication Package

This folder is a clean code and data replication package for the paper:

**Validating ETF Proxies for Fama French Style Factors: Evidence from Classification Based Trading Strategies**

It contains the minimum data, code, and selected outputs needed to reproduce the empirical results used in the manuscript. The manuscript source itself is intentionally not included in this repository.

## Folder Structure

```text
replication_package/
  code/
    00_prepare_data.py
    01_publication_analysis.py
    02_factor_proxy_definition_horse_race.py
    03_plot_weekly_knn_cost_panels.py
    04_plot_weekly_logit_winners_losers_cost_panels.py
    05_plot_winners_losers_k_sharpe.py
    06_make_payoff_asymmetry_table.py
    run_replication.py
  data/
    ff5_cleaned_daily.csv
    etf_daily_returns.csv
    alt_etf_daily_returns.csv
  outputs/
    selected paper figures and generated tables
  requirements.txt
```

## Data Sources and Inputs

The replication uses daily Fama French five factor returns and daily ETF returns.

The Fama French data come from the Kenneth R. French Data Library, specifically the daily Fama French five factor file. The cleaned file included here is `data/ff5_cleaned_daily.csv`. The cleaning step parses the French library CSV, converts percentage returns to decimal returns, renames `Mkt-RF` to `MKT`, and keeps `MKT`, `SMB`, `HML`, `RMW`, `CMA`, and `RF`.

The ETF data are downloaded from Yahoo Finance through the `yfinance` Python package using adjusted prices with `auto_adjust=True`. Daily ETF returns are computed as adjusted close percentage changes. The included cleaned ETF files are:

```text
data/etf_daily_returns.csv
data/alt_etf_daily_returns.csv
```

The ETF proxies in the paper are:

```text
SMB = IJR - SPY
HML = SPYV - SPYG
RMW = QUAL - ARKK
CMA = SPHQ - QQQ
```

The common empirical sample starts after the latest required ETF becomes available, so the analysis avoids filling unavailable ETF histories.

When using this repository, cite the Kenneth R. French Data Library for the Fama French factor returns and Yahoo Finance as accessed through `yfinance` for ETF adjusted price histories. The ETF data are redistributed here only as cleaned research inputs for replication.

## Rebuild Clean Data

The cleaned data files are included so the replication can run without network access. To rebuild them from source, place the raw French daily five factor CSV at:

```text
data/F-F_Research_Data_5_Factors_2x3_daily.csv
```

Then run:

```bash
python code/00_prepare_data.py
```

This script cleans the French daily factor file and downloads ETF adjusted prices from Yahoo Finance through `yfinance`. Rebuilt ETF data may differ slightly from the included files if Yahoo Finance revises historical adjusted prices.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Results

Run the full pipeline:

```bash
python code/run_replication.py
```

The pipeline writes CSV tables, figures, and LaTeX table snippets to `outputs/`.

## Main Scripts

`00_prepare_data.py` rebuilds the cleaned data inputs from the Kenneth R. French daily five factor file and Yahoo Finance ETF adjusted prices. This step is optional because the cleaned input CSVs are included.

`01_publication_analysis.py` builds the core proxy validation tables, daily and weekly strategy panels, Winners and Losers strategies, KNN strategies, logistic strategies, out of sample tests, and baseline figures.

`02_factor_proxy_definition_horse_race.py` evaluates alternative A minus B ETF proxy definitions for factor validation.

`03_plot_weekly_knn_cost_panels.py` creates the weekly KNN wealth plots at 0, 1, and 2 basis points per switch.

`04_plot_weekly_logit_winners_losers_cost_panels.py` creates weekly logistic, Winners, and Losers wealth plots at 0, 1, and 2 basis points per switch.

`05_plot_winners_losers_k_sharpe.py` creates the weekly Winners and Losers k sensitivity diagnostic.

`06_make_payoff_asymmetry_table.py` regenerates the payoff asymmetry and directional accuracy table.

## Notes

The broader working project contains many exploratory notebooks, draft scripts, manuscript files, and old output files. They are intentionally excluded from this replication package so that reviewers can rerun the empirical results from a small and auditable code base.
