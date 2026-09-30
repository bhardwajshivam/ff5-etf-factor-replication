import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BASELINE_COST_BPS = 2


def main():
    summary = pd.read_csv(OUTPUT_DIR / "paper_weekly_strategy_summary.csv")
    subset = summary[
        (summary["transaction_cost_bps"] == BASELINE_COST_BPS)
        & (summary["strategy_type"].isin(["Winners", "Losers"]))
    ].copy()

    factors = ["CMA", "HML", "RMW", "SMB"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True)
    axes = axes.ravel()

    selected_rows = []
    for ax, factor in zip(axes, factors):
        factor_data = subset[subset["factor"] == factor].copy()
        for strategy_type, color in [("Winners", "#1f77b4"), ("Losers", "#d62728")]:
            line_data = factor_data[factor_data["strategy_type"] == strategy_type].sort_values("k")
            ax.plot(
                line_data["k"],
                line_data["sharpe"],
                marker="o",
                linewidth=2.0,
                color=color,
                label=strategy_type,
            )
            best = line_data.sort_values(["sharpe", "final_wealth"], ascending=False).iloc[0]
            selected_rows.append(
                {
                    "factor": factor,
                    "strategy_type": strategy_type,
                    "selected_k": int(best["k"]),
                    "strategy_name": best["strategy_name"],
                    "sharpe": best["sharpe"],
                    "final_wealth": best["final_wealth"],
                }
            )
            ax.scatter(best["k"], best["sharpe"], s=80, color=color, edgecolor="black", zorder=5)
            ax.annotate(
                f"k={int(best['k'])}",
                xy=(best["k"], best["sharpe"]),
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_title(factor)
        ax.set_ylabel("Sharpe ratio")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(sorted(subset["k"].dropna().unique()))

    axes[-2].set_xlabel("Lookback window k")
    axes[-1].set_xlabel("Lookback window k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Weekly Winners and Losers Sharpe Ratio by k, Cost={BASELINE_COST_BPS} bps", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = OUTPUT_DIR / "paper_weekly_winners_losers_k_sharpe.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUTPUT_DIR / "paper_weekly_winners_losers_selected_k_by_sharpe.csv", index=False)

    print(f"Saved plot: {output_path}")
    print(f"Saved selected k table: {OUTPUT_DIR / 'paper_weekly_winners_losers_selected_k_by_sharpe.csv'}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
