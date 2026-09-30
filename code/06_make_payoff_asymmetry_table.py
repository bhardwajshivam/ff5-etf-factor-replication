from math import erf, sqrt
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_PATH = OUTPUT_DIR / "payoff_asymmetry_table.tex"
CSV_PATH = OUTPUT_DIR / "paper_payoff_asymmetry_table.csv"


STRATEGIES = [
    ("Daily CMA Losers (k = 13)", "paper_strategy_returns.csv", "CMA-13L"),
    ("Daily HML Logit", "paper_logistic_strategy_returns.csv", "HML-Logit"),
    ("Daily HML KNN15", "paper_knn_strategy_returns.csv", "HML-KNN15"),
    ("Daily RMW KNN11", "paper_knn_strategy_returns.csv", "RMW-KNN11"),
    ("Daily CMA KNN3", "paper_knn_strategy_returns.csv", "CMA-KNN3"),
    ("Daily SMB KNN3", "paper_knn_strategy_returns.csv", "SMB-KNN3"),
    ("Weekly HML Winners (k = 13)", "paper_weekly_strategy_returns.csv", "HML-13W"),
    ("Weekly CMA Losers (k = 13)", "paper_weekly_strategy_returns.csv", "CMA-13L"),
    ("Weekly CMA KNN23", "paper_weekly_knn_strategy_returns.csv", "CMA-KNN23"),
    ("Weekly HML KNN15", "paper_weekly_knn_strategy_returns.csv", "HML-KNN15"),
    ("Weekly RMW KNN11", "paper_weekly_knn_strategy_returns.csv", "RMW-KNN11"),
    ("Weekly SMB KNN23", "paper_weekly_knn_strategy_returns.csv", "SMB-KNN23"),
]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def summarize_strategy(label: str, file_name: str, strategy_name: str) -> dict:
    data = pd.read_csv(OUTPUT_DIR / file_name)
    subset = data.loc[data["strategy_name"].eq(strategy_name)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for {strategy_name} in {file_name}")

    actual_up = subset["true_label"].eq(1)
    predicted_up = subset["predicted_label"].eq(1)
    accuracy = subset["correct"].astype(float).mean()
    expected_accuracy = (
        actual_up.mean() * predicted_up.mean()
        + (1.0 - actual_up.mean()) * (1.0 - predicted_up.mean())
    )
    variance = expected_accuracy * (1.0 - expected_accuracy) / len(subset)
    pt_stat = (
        (accuracy - expected_accuracy) / sqrt(variance)
        if variance > 0
        else float("nan")
    )
    one_sided_p = 1.0 - normal_cdf(pt_stat)

    correct = subset["correct"].astype(bool)
    mu_plus = subset.loc[correct, "proxy_return"].abs().mean()
    mu_minus = subset.loc[~correct, "proxy_return"].abs().mean()
    break_even_accuracy = mu_minus / (mu_plus + mu_minus)
    timing_bps = (0.5 * subset["predicted_label"] * subset["proxy_return"]).mean() * 10000

    return {
        "Strategy": label,
        "Accuracy": accuracy,
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
        "p_star": break_even_accuracy,
        "Timing_bps_per_period": timing_bps,
        "PT_stat": pt_stat,
        "PT_p_one_sided": one_sided_p,
        "Observations": len(subset),
    }


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}\\%"


def fmt_return_pct(value: float) -> str:
    return f"{value * 100:.3f}\\%"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_latex(rows: pd.DataFrame) -> str:
    body = []
    for _, row in rows.iterrows():
        pt_cell = f"{fmt_float(row['PT_stat'])} ({row['PT_p_one_sided']:.3f})"
        body.append(
            " & ".join(
                [
                    row["Strategy"],
                    fmt_pct(row["Accuracy"]),
                    fmt_return_pct(row["mu_plus"]),
                    fmt_return_pct(row["mu_minus"]),
                    fmt_pct(row["p_star"]),
                    fmt_float(row["Timing_bps_per_period"]),
                    pt_cell,
                ]
            )
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Payoff asymmetry and directional accuracy in selected strategies}",
            r"\label{tab:payoff_asymmetry}",
            r"\scriptsize",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Strategy & Accuracy & $\mu^{+}$ & $\mu^{-}$ & $p^{\ast}$ & Timing & PT Stat. (p) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\begin{flushleft}",
            r"\footnotesize{\textit{Notes:} $\mu^{+}$ and $\mu^{-}$ are the average absolute ETF proxy spreads $|F^{ETF}_{j,t}|$ in correctly and incorrectly classified periods. $p^{\ast}=\mu^{-}/(\mu^{+}+\mu^{-})$ is the break-even accuracy. Timing is the average gross timing component $\frac{1}{2}d_{j,t}F^{ETF}_{j,t}$, in basis points per day or per week. PT Stat. is the Pesaran and Timmermann directional statistic with its one-sided p-value. All strategy returns are net of 2 basis points per switch, while the timing component is shown before costs.}",
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    rows = pd.DataFrame(
        [summarize_strategy(*strategy) for strategy in STRATEGIES]
    )
    rows.to_csv(CSV_PATH, index=False)
    TABLE_PATH.write_text(build_latex(rows), encoding="utf-8")
    print(f"Saved {CSV_PATH}")
    print(f"Saved {TABLE_PATH}")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
