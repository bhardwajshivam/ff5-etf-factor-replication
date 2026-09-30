# ETF Factor Proxy Replication Package

This folder is a clean replication package for the paper:

**Validating ETF Proxies for Fama French Style Factors: Evidence from Classification Based Trading Strategies**

It contains the minimum data, code, paper files, and selected outputs needed to reproduce the empirical results used in the manuscript.

## Folder Structure

```text
replication_package/
  code/
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
  paper/
    paper.tex
    citations.bib
  requirements.txt
```

## Data Inputs

The replication uses daily Fama French five factor returns and daily ETF returns. The ETF proxies in the paper are:

```text
SMB = IJR - SPY
HML = SPYV - SPYG
RMW = QUAL - ARKK
CMA = SPHQ - QQQ
```

The common empirical sample starts after the latest required ETF becomes available, so the analysis avoids filling unavailable ETF histories.

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

The pipeline writes CSV tables, figures, and LaTeX table snippets to `outputs/` and `paper/`.

## Main Scripts

`01_publication_analysis.py` builds the core proxy validation tables, daily and weekly strategy panels, Winners and Losers strategies, KNN strategies, logistic strategies, out of sample tests, and baseline figures.

`02_factor_proxy_definition_horse_race.py` evaluates alternative A minus B ETF proxy definitions for factor validation.

`03_plot_weekly_knn_cost_panels.py` creates the weekly KNN wealth plots at 0, 1, and 2 basis points per switch.

`04_plot_weekly_logit_winners_losers_cost_panels.py` creates weekly logistic, Winners, and Losers wealth plots at 0, 1, and 2 basis points per switch.

`05_plot_winners_losers_k_sharpe.py` creates the weekly Winners and Losers k sensitivity diagnostic.

`06_make_payoff_asymmetry_table.py` regenerates the payoff asymmetry and directional accuracy table.

## Paper Compilation

The manuscript is in `paper/paper.tex`. It uses the MDPI LaTeX class:

```latex
\documentclass[investment,article,submit,pdftex,moreauthors]{Definitions/mdpi}
```

To compile the paper, place the official MDPI `Definitions/` folder next to `paper.tex`, or compile within the MDPI template directory after copying `paper.tex`, `citations.bib`, and `outputs/`.

## Notes

The broader working project contains many exploratory notebooks, draft scripts, and old output files. They are intentionally excluded from this replication package so that reviewers can rerun the paper results from a small and auditable code base.
