import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
INITIAL_WEALTH = 100.0
PRIMARY_K = 13
COMMON_EVALUATION_START = pd.Timestamp("2016-01-01")


PANEL_CONFIGS = [
    {
        "panel_name": "weekly_logit",
        "title_name": "Weekly Logit",
        "returns_file": "paper_weekly_logistic_strategy_returns.csv",
        "strategies": {
            "CMA": "CMA-Logit",
            "HML": "HML-Logit",
            "RMW": "RMW-Logit",
            "SMB": "SMB-Logit",
        },
    },
    {
        "panel_name": "weekly_winners_k13",
        "title_name": "Weekly Winners k=13",
        "returns_file": "paper_weekly_strategy_returns.csv",
        "strategies": {
            "CMA": "CMA-13W",
            "HML": "HML-13W",
            "RMW": "RMW-13W",
            "SMB": "SMB-13W",
        },
    },
    {
        "panel_name": "weekly_losers_k13",
        "title_name": "Weekly Losers k=13",
        "returns_file": "paper_weekly_strategy_returns.csv",
        "strategies": {
            "CMA": "CMA-13L",
            "HML": "HML-13L",
            "RMW": "RMW-13L",
            "SMB": "SMB-13L",
        },
    },
]


def load_etf_returns():
    base = pd.read_csv(DATA_DIR / "etf_daily_returns.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    alt = pd.read_csv(DATA_DIR / "alt_etf_daily_returns.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    base.index.name = "date"
    alt.index.name = "date"
    return alt.combine_first(base).apply(pd.to_numeric, errors="coerce")


def wealth_index(returns):
    returns = returns.dropna()
    return INITIAL_WEALTH * (1.0 + returns).cumprod()


def make_weekly_spy(etf_returns):
    return (1.0 + etf_returns["SPY"].dropna()).resample("W-FRI").prod() - 1.0


def plot_panel_for_cost(panel_config, strategy_returns, weekly_spy, bps):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    summary_rows = []

    for factor, strategy_name in panel_config["strategies"].items():
        subset = strategy_returns[
            (strategy_returns["strategy_name"] == strategy_name)
            & (strategy_returns["date"] >= COMMON_EVALUATION_START)
        ].copy()
        if subset.empty:
            raise ValueError(f"No rows found for {strategy_name}")
        subset = subset.sort_values("date")
        dates = pd.DatetimeIndex(subset["date"])
        net_returns = subset["chosen_return_before_cost"] - subset["position_switch"] * bps / 10000.0
        wealth = wealth_index(pd.Series(net_returns.to_numpy(), index=dates))
        ax.plot(wealth.index, wealth.values, linewidth=2.0, label=f"{strategy_name} ({wealth.iloc[-1]:.1f})")
        summary_rows.append(
            {
                "panel": panel_config["panel_name"],
                "bps": bps,
                "factor": factor,
                "strategy_name": strategy_name,
                "final_wealth": wealth.iloc[-1],
                "switches": int(subset["position_switch"].sum()),
            }
        )

    selected_names = set(panel_config["strategies"].values())
    all_dates = pd.DatetimeIndex(
        strategy_returns[
            (strategy_returns["strategy_name"].isin(selected_names))
            & (strategy_returns["date"] >= COMMON_EVALUATION_START)
        ]["date"]
        .sort_values()
        .unique()
    )
    spy = weekly_spy.reindex(all_dates)
    spy_wealth = wealth_index(pd.Series(spy.to_numpy(), index=all_dates))
    ax.plot(
        spy_wealth.index,
        spy_wealth.values,
        color="black",
        linewidth=1.8,
        linestyle="--",
        label=f"SPY ({spy_wealth.iloc[-1]:.1f})",
    )

    ax.set_title(f"{panel_config['title_name']} Strategy Wealth vs SPY, Cost={bps} bps")
    ax.set_ylabel("Wealth index, initial value = 100")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"paper_{panel_config['panel_name']}_primary_strategy_bps_{bps}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path, summary_rows


def main():
    etf_returns = load_etf_returns()
    weekly_spy = make_weekly_spy(etf_returns)

    output_paths = []
    rows = []
    cache = {}

    for config in PANEL_CONFIGS:
        returns_file = config["returns_file"]
        if returns_file not in cache:
            cache[returns_file] = pd.read_csv(OUTPUT_DIR / returns_file, parse_dates=["date"])
        strategy_returns = cache[returns_file]
        for bps in [0, 1, 2]:
            out_path, summary_rows = plot_panel_for_cost(config, strategy_returns, weekly_spy, bps)
            output_paths.append(out_path)
            rows.extend(summary_rows)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "paper_weekly_logit_winners_losers_bps_summary.csv", index=False)

    print("Generated weekly logit, winners, and losers cost plots:")
    for path in output_paths:
        print(path)
    print(f"Saved summary: {OUTPUT_DIR / 'paper_weekly_logit_winners_losers_bps_summary.csv'}")


if __name__ == "__main__":
    main()
