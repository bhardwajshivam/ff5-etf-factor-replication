import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PREFIX = "paper_proxy_horse_race"

FF5_PATH = DATA_DIR / "ff5_cleaned_daily.csv"
ALT_ETF_RETURNS_PATH = DATA_DIR / "alt_etf_daily_returns.csv"
BASE_ETF_RETURNS_PATH = DATA_DIR / "etf_daily_returns.csv"

COMMON_SAMPLE_START = pd.Timestamp("2014-08-01")
PERIODS_PER_YEAR = 252
ROLLING_WINDOW_DAYS = 252 * 3
MIN_YEAR_OBS = 80

CURRENT_DEFINITIONS = {
    "RMW": ("QUAL", "ARKK"),
    "CMA": ("SPLV", "SPHB"),
}

PROXY_CANDIDATES = {
    "RMW": [
        ("QUAL", "ARKK", "current quality minus speculative innovation"),
        ("QUAL", "SPY", "quality tilt versus broad market"),
        ("SPHQ", "SPY", "S&P quality tilt versus broad market"),
        ("QUAL", "SPHB", "quality minus high beta"),
        ("SPHQ", "SPHB", "S&P quality minus high beta"),
        ("QUAL", "SPYG", "quality minus growth"),
        ("SPHQ", "SPYG", "S&P quality minus growth"),
    ],
    "CMA": [
        ("SPLV", "SPHB", "current low volatility minus high beta"),
        ("SPLV", "SPYG", "low volatility minus growth"),
        ("SPLV", "QQQ", "low volatility minus Nasdaq 100"),
        ("SPLV", "SPY", "low volatility tilt versus broad market"),
        ("SPHQ", "QQQ", "quality minus Nasdaq 100"),
        ("SPHQ", "SPYG", "quality minus growth"),
        ("SPYV", "SPYG", "value minus growth benchmark"),
    ],
}


def load_inputs():
    ff5 = pd.read_csv(FF5_PATH, parse_dates=["date"]).set_index("date").sort_index()
    alt = pd.read_csv(ALT_ETF_RETURNS_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    base = pd.read_csv(BASE_ETF_RETURNS_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    alt.index.name = "date"
    base.index.name = "date"
    etf_returns = alt.combine_first(base).apply(pd.to_numeric, errors="coerce")
    return ff5[["SMB", "HML", "RMW", "CMA"]], etf_returns


def ols(y, x):
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 10:
        return None
    x_design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy()])
    y_values = frame["y"].to_numpy()
    beta = np.linalg.lstsq(x_design, y_values, rcond=None)[0]
    fitted = x_design @ beta
    residual = y_values - fitted
    dof = len(frame) - x_design.shape[1]
    sigma2 = float((residual @ residual) / dof) if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(x_design.T @ x_design)
    se = np.sqrt(np.diag(cov))
    ss_total = float(((y_values - y_values.mean()) ** 2).sum())
    ss_resid = float((residual**2).sum())
    return {
        "alpha": beta[0],
        "beta": beta[1],
        "alpha_tstat": beta[0] / se[0] if se[0] else np.nan,
        "beta_tstat": beta[1] / se[1] if se[1] else np.nan,
        "r_squared": 1.0 - ss_resid / ss_total if ss_total else np.nan,
        "residual": pd.Series(residual, index=frame.index),
        "fitted": pd.Series(fitted, index=frame.index),
        "nobs": len(frame),
    }


def candidate_returns(etf_returns, leg_a, leg_b):
    missing = [ticker for ticker in [leg_a, leg_b] if ticker not in etf_returns.columns]
    if missing:
        return None
    return etf_returns[leg_a] - etf_returns[leg_b]


def evaluate_candidates(ff5, etf_returns):
    rows = []
    yearly_rows = []
    rolling_rows = []
    fit_rows = []

    for factor, candidates in PROXY_CANDIDATES.items():
        for leg_a, leg_b, description in candidates:
            proxy = candidate_returns(etf_returns, leg_a, leg_b)
            if proxy is None:
                continue
            frame = pd.concat(
                [ff5[factor].rename("official"), proxy.rename("proxy")],
                axis=1,
            )
            frame = frame[frame.index >= COMMON_SAMPLE_START].dropna()
            result = ols(frame["official"], frame["proxy"])
            if result is None:
                continue
            residual = result["residual"]
            raw_error = frame["official"] - frame["proxy"]
            row = {
                "factor": factor,
                "candidate": f"{leg_a} - {leg_b}",
                "leg_a": leg_a,
                "leg_b": leg_b,
                "description": description,
                "is_current_definition": (leg_a, leg_b) == CURRENT_DEFINITIONS.get(factor),
                "date_start": frame.index.min().date(),
                "date_end": frame.index.max().date(),
                "observations": len(frame),
                "correlation": frame["official"].corr(frame["proxy"]),
                "abs_correlation": abs(frame["official"].corr(frame["proxy"])),
                "sign_match_rate": (np.sign(frame["official"]) == np.sign(frame["proxy"])).mean(),
                "alpha_daily": result["alpha"],
                "alpha_annualized": result["alpha"] * PERIODS_PER_YEAR,
                "beta": result["beta"],
                "beta_tstat": result["beta_tstat"],
                "r_squared": result["r_squared"],
                "raw_tracking_error": raw_error.std() * np.sqrt(PERIODS_PER_YEAR),
                "residual_volatility": residual.std() * np.sqrt(PERIODS_PER_YEAR),
            }
            row["rank_score"] = (
                row["r_squared"]
                + row["abs_correlation"]
                + row["sign_match_rate"]
                - 0.25 * row["raw_tracking_error"]
            )
            rows.append(row)

            fit = pd.DataFrame(
                {
                    "date": frame.index,
                    "factor": factor,
                    "candidate": f"{leg_a} - {leg_b}",
                    "official_return": frame["official"].to_numpy(),
                    "proxy_return": frame["proxy"].to_numpy(),
                    "fitted_return": result["fitted"].reindex(frame.index).to_numpy(),
                    "residual": result["residual"].reindex(frame.index).to_numpy(),
                }
            )
            fit_rows.append(fit)

            for year, group in frame.groupby(frame.index.year):
                if len(group) < MIN_YEAR_OBS:
                    continue
                year_result = ols(group["official"], group["proxy"])
                if year_result is None:
                    continue
                yearly_rows.append(
                    {
                        "factor": factor,
                        "candidate": f"{leg_a} - {leg_b}",
                        "year": year,
                        "observations": len(group),
                        "correlation": group["official"].corr(group["proxy"]),
                        "sign_match_rate": (np.sign(group["official"]) == np.sign(group["proxy"])).mean(),
                        "beta": year_result["beta"],
                        "r_squared": year_result["r_squared"],
                        "raw_tracking_error": (group["official"] - group["proxy"]).std() * np.sqrt(PERIODS_PER_YEAR),
                    }
                )

            for end_position in range(ROLLING_WINDOW_DAYS, len(frame) + 1):
                window = frame.iloc[end_position - ROLLING_WINDOW_DAYS:end_position]
                rolling_result = ols(window["official"], window["proxy"])
                if rolling_result is None:
                    continue
                rolling_rows.append(
                    {
                        "factor": factor,
                        "candidate": f"{leg_a} - {leg_b}",
                        "date": window.index[-1],
                        "rolling_correlation": window["official"].corr(window["proxy"]),
                        "rolling_r_squared": rolling_result["r_squared"],
                        "rolling_beta": rolling_result["beta"],
                        "rolling_tracking_error": (window["official"] - window["proxy"]).std()
                        * np.sqrt(PERIODS_PER_YEAR),
                    }
                )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["factor", "rank_score"], ascending=[True, False])
    return (
        summary,
        pd.DataFrame(yearly_rows),
        pd.DataFrame(rolling_rows),
        pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame(),
    )


def plot_scorecard(summary):
    for factor, group in summary.groupby("factor"):
        group = group.sort_values("r_squared", ascending=True)
        labels = group["candidate"].tolist()
        y = np.arange(len(group))
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(y - 0.2, group["r_squared"], height=0.2, label="R-squared")
        ax.barh(y, group["abs_correlation"], height=0.2, label="Absolute correlation")
        ax.barh(y + 0.2, group["sign_match_rate"], height=0.2, label="Sign match")
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title(f"{factor} ETF Proxy Candidate Validation")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_{factor.lower()}_scorecard.png", dpi=200)
        plt.close()


def plot_rolling_best(rolling, summary):
    best_candidates = summary.sort_values("rank_score", ascending=False).groupby("factor").head(3)
    for factor, best in best_candidates.groupby("factor"):
        subset = rolling[
            (rolling["factor"] == factor)
            & (rolling["candidate"].isin(best["candidate"]))
        ]
        plt.figure(figsize=(12, 6))
        for candidate, group in subset.groupby("candidate"):
            group = group.sort_values("date")
            plt.plot(group["date"], group["rolling_r_squared"], label=candidate, linewidth=1.8)
        plt.ylim(0, 1)
        plt.title(f"{factor} Rolling 3-Year R-squared, Top ETF Proxy Candidates")
        plt.xlabel("Date")
        plt.ylabel("Rolling R-squared")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_{factor.lower()}_rolling_top3_r2.png", dpi=200)
        plt.close()


def write_method_note(summary):
    lines = [
        "ETF proxy definition horse race",
        "",
        "Purpose:",
        "  Compare only A-minus-B ETF proxy definitions for RMW and CMA.",
        "  Candidates are selected by proxy validity against official Fama-French factors, not by strategy performance.",
        "",
        "Ranking metrics:",
        "  R-squared from official factor on ETF proxy regression.",
        "  Absolute correlation with official factor.",
        "  Sign match rate.",
        "  Raw tracking error penalty.",
        "",
        "Best candidate by factor:",
    ]
    best = summary.sort_values("rank_score", ascending=False).groupby("factor").head(1).sort_values("factor")
    for _, row in best.iterrows():
        lines.append(
            f"  {row['factor']}: {row['candidate']} "
            f"(R2={row['r_squared']:.3f}, corr={row['correlation']:.3f}, "
            f"sign={row['sign_match_rate']:.3f}, TE={row['raw_tracking_error']:.3f})"
        )
    (OUTPUT_DIR / f"{OUTPUT_PREFIX}_method_note.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ff5, etf_returns = load_inputs()
    summary, yearly, rolling, fit = evaluate_candidates(ff5, etf_returns)

    summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv", index=False)
    yearly.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_yearly.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_rolling_3y.csv", index=False)
    fit.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_actual_fitted.csv", index=False)
    plot_scorecard(summary)
    plot_rolling_best(rolling, summary)
    write_method_note(summary)

    print("Saved proxy definition horse race outputs to outputs/")
    print("\nRanking by factor:")
    cols = [
        "factor",
        "candidate",
        "is_current_definition",
        "observations",
        "correlation",
        "sign_match_rate",
        "beta",
        "r_squared",
        "raw_tracking_error",
        "residual_volatility",
        "rank_score",
    ]
    print(summary[cols].to_string(index=False))
    print("\nRecommended A-minus-B definitions:")
    best = summary.sort_values("rank_score", ascending=False).groupby("factor").head(1).sort_values("factor")
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
