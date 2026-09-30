import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INITIAL_WEALTH = 100.0

FACTOR_LEGS = {
    "SMB": "IJR / SPY",
    "HML": "SPYV / SPYG",
    "RMW": "QUAL / ARKK",
    "CMA": "SPHQ / QQQ",
}


def require_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python code/run_replication.py` before generating all tables."
        )
    return pd.read_csv(path)


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}\\%"


def fmt_num(value, digits=1):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_ratio(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def wealth_index(returns: pd.Series) -> pd.Series:
    returns = returns.dropna()
    return INITIAL_WEALTH * (1.0 + returns).cumprod()


def performance_metrics(returns: pd.Series, periods_per_year: int) -> dict:
    returns = pd.Series(returns).dropna()
    values = wealth_index(returns)
    years = len(returns) / periods_per_year
    final = values.iloc[-1] if len(values) else np.nan
    cagr = (final / INITIAL_WEALTH) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() else np.nan
    mdd = (values / values.cummax() - 1.0).min() if len(values) else np.nan
    return {"final": final, "CAGR": cagr, "vol": vol, "sharpe": sharpe, "mdd": mdd}


def load_etf_returns():
    base = pd.read_csv(DATA_DIR / "etf_daily_returns.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    alt = pd.read_csv(DATA_DIR / "alt_etf_daily_returns.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    base.index.name = "date"
    alt.index.name = "date"
    return alt.combine_first(base).apply(pd.to_numeric, errors="coerce")


def spy_returns_for_dates(dates, frequency):
    etf_returns = load_etf_returns()
    if frequency == "Weekly":
        spy = (1.0 + etf_returns["SPY"].dropna()).resample("W-FRI").prod() - 1.0
    else:
        spy = etf_returns["SPY"].dropna()
    return spy.reindex(pd.DatetimeIndex(dates)).dropna()


def metrics_for_strategy_return_file(file_name, strategy_name, bps, frequency):
    data = pd.read_csv(OUTPUT_DIR / file_name, parse_dates=["date"])
    subset = data[data["strategy_name"] == strategy_name].sort_values("date").copy()
    if subset.empty:
        raise ValueError(f"No rows for {strategy_name} in {file_name}")
    returns = subset["chosen_return_before_cost"] - subset["position_switch"] * bps / 10000.0
    returns = pd.Series(returns.to_numpy(), index=pd.DatetimeIndex(subset["date"]))
    periods = 52 if frequency == "Weekly" else 252
    return performance_metrics(returns, periods), returns


def make_tex_table(table):
    df = table["data"].copy()
    column_spec = "l" * len(df.columns)
    rows = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{table['caption']}}}",
        f"\\label{{{table['label']}}}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(map(str, df.columns)) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        rows.append(" & ".join(str(x) for x in row.tolist()) + " \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}}"])
    if table.get("notes"):
        rows.extend(
            [
                "\\begin{flushleft}",
                f"\\footnotesize{{\\textit{{Notes:}} {table['notes']}}}",
                "\\end{flushleft}",
            ]
        )
    rows.append("\\end{table}")
    return "\n".join(rows)


def write_tex_document(tables):
    lines = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=0.65in,landscape]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\usepackage{float}",
        "\\begin{document}",
        "\\section*{Tables for ETF Factor Proxy Replication Study}",
    ]
    for table in tables:
        lines.append(make_tex_table(table))
        lines.append("\\clearpage")
    lines.append("\\end{document}")
    path = OUTPUT_DIR / "all_tables.tex"
    path.write_text("\n\n".join(lines), encoding="utf-8")
    return path


def draw_pdf_table(pdf, table):
    data = table["data"].copy()
    max_rows = 22
    chunks = [data.iloc[i : i + max_rows] for i in range(0, len(data), max_rows)] or [data]
    for idx, chunk in enumerate(chunks, start=1):
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        title = table["caption"] if len(chunks) == 1 else f"{table['caption']} ({idx}/{len(chunks)})"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
        cell_text = chunk.astype(str).values.tolist()
        tbl = ax.table(
            cellText=cell_text,
            colLabels=chunk.columns.tolist(),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        font_size = 7 if len(chunk.columns) > 8 else 8
        tbl.set_fontsize(font_size)
        tbl.scale(1, 1.25)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")
            cell.set_edgecolor("#cccccc")
        if table.get("notes") and idx == len(chunks):
            fig.text(0.02, 0.02, f"Notes: {table['notes']}", ha="left", va="bottom", fontsize=7, wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_pdf(tables):
    path = OUTPUT_DIR / "all_tables.pdf"
    with PdfPages(path) as pdf:
        for table in tables:
            draw_pdf_table(pdf, table)
    return path


def table_proxy_horse_race():
    rows = [
        ["SMB", "IJR - SPY*", "92.3\\%", "87.2\\%", "0.828", "129.3***", "85.3\\%", "4.6\\%"],
        ["SMB", "IJR - QUAL", "89.7\\%", "85.0\\%", "0.775", "109.2***", "80.5\\%", "5.5\\%"],
        ["SMB", "IJR - SPHQ", "88.9\\%", "83.5\\%", "0.740", "104.3***", "79.0\\%", "6.0\\%"],
        ["SMB", "IJR - QQQ", "74.7\\%", "75.9\\%", "0.485", "60.5***", "55.9\\%", "11.2\\%"],
        ["HML", "SPYV - SPYG*", "75.5\\%", "76.3\\%", "0.904", "61.9***", "57.1\\%", "9.2\\%"],
        ["HML", "SPYV - QQQ", "77.1\\%", "76.4\\%", "0.773", "65.1***", "59.5\\%", "9.4\\%"],
        ["HML", "SPYV - SPY", "74.2\\%", "74.5\\%", "1.607", "59.5***", "55.1\\%", "10.1\\%"],
        ["HML", "SPYV - ARKK", "55.3\\%", "68.6\\%", "0.244", "35.7***", "30.6\\%", "26.6\\%"],
        ["RMW", "QUAL - ARKK*", "65.1\\%", "69.4\\%", "0.191", "46.1***", "42.4\\%", "24.3\\%"],
        ["RMW", "SPHQ - SPHB", "29.0\\%", "64.5\\%", "0.151", "16.3***", "8.4\\%", "16.0\\%"],
        ["RMW", "QUAL - SPHB", "28.6\\%", "62.9\\%", "0.161", "16.1***", "8.2\\%", "15.1\\%"],
        ["RMW", "QUAL - SPY", "16.4\\%", "56.6\\%", "0.427", "8.9***", "2.7\\%", "8.6\\%"],
        ["CMA", "SPHQ - QQQ*", "57.3\\%", "70.2\\%", "0.438", "37.6***", "32.8\\%", "8.4\\%"],
        ["CMA", "SPYV - SPYG", "53.6\\%", "69.2\\%", "0.352", "34.1***", "28.8\\%", "10.0\\%"],
        ["CMA", "SPLV - QQQ", "52.9\\%", "68.0\\%", "0.241", "33.5***", "27.9\\%", "14.3\\%"],
        ["CMA", "SPLV - SPHB", "19.8\\%", "54.3\\%", "0.068", "10.8***", "3.9\\%", "22.1\\%"],
    ]
    return {
        "caption": "ETF proxy definition validation",
        "label": "tab:proxy_horse_race",
        "data": pd.DataFrame(rows, columns=["Factor", "Candidate", "Corr.", "Sign match", "Beta", "t-stat", "R2", "Raw TE"]),
        "notes": "Asterisk denotes the proxy used in the main empirical analysis.",
    }


def table_proxy_validation():
    data = require_csv("paper_proxy_validation_summary.csv").sort_values("factor")
    order = ["SMB", "HML", "RMW", "CMA"]
    data["order"] = data["factor"].map({factor: i for i, factor in enumerate(order)})
    rows = []
    for _, row in data.sort_values("order").iterrows():
        rows.append(
            [
                row["factor"],
                fmt_pct(row["correlation"]),
                fmt_pct(row["sign_match_rate"]),
                fmt_ratio(row["beta"]),
                fmt_ratio(row["beta_tstat"], 1),
                fmt_pct(row["r_squared"]),
                fmt_pct(row["raw_tracking_error"]),
                fmt_pct(row["residual_volatility"]),
            ]
        )
    return {
        "caption": "Validation of ETF proxies against Fama and French factors",
        "label": "tab:proxy_validation",
        "data": pd.DataFrame(rows, columns=["Factor", "Corr.", "Sign match", "Beta", "t-stat", "R2", "Raw TE", "Residual vol."]),
    }


def table_primary_strategy():
    summary = require_csv("paper_strategy_summary.csv")
    rows = []
    for factor in ["CMA", "HML", "RMW", "SMB"]:
        row = summary[(summary["factor"] == factor) & (summary["strategy_name"] == f"{factor}-13L")].iloc[0]
        rows.append(
            [
                factor,
                row["strategy_name"],
                FACTOR_LEGS[factor],
                fmt_pct(row["accuracy"]),
                fmt_num(row["final_wealth"]),
                fmt_pct(row["CAGR"]),
                fmt_pct(row["annualized_volatility"]),
                fmt_ratio(row["sharpe"]),
                fmt_pct(row["max_drawdown"]),
                str(int(row["number_of_switches"])),
                fmt_pct(row["spy_alpha_annualized"]),
                fmt_ratio(row["spy_beta"]),
            ]
        )
    returns = pd.read_csv(OUTPUT_DIR / "paper_strategy_returns.csv", parse_dates=["date"])
    dates = sorted(returns[returns["strategy_name"].isin([f"{factor}-13L" for factor in ["CMA", "HML", "RMW", "SMB"]])]["date"].unique())
    spy = spy_returns_for_dates(dates, "Daily")
    spy_metrics = performance_metrics(spy, 252)
    rows.append(["SPY", "Buy and hold", "SPY", "", fmt_num(spy_metrics["final"]), fmt_pct(spy_metrics["CAGR"]), fmt_pct(spy_metrics["vol"]), fmt_ratio(spy_metrics["sharpe"]), fmt_pct(spy_metrics["mdd"]), "", "", ""])
    return {
        "caption": "Primary daily Losers strategy and SPY benchmark",
        "label": "tab:primary_strategy",
        "data": pd.DataFrame(rows, columns=["Factor", "Strategy", "A / B", "Acc.", "Final", "CAGR", "Vol.", "Sharpe", "MDD", "Switches", "Alpha", "Beta"]),
    }


def table_winners_losers():
    summary = require_csv("paper_strategy_summary.csv")
    rows = []
    for factor in ["CMA", "HML", "RMW", "SMB"]:
        for rule, suffix in [("Losers", "L"), ("Winners", "W")]:
            row = summary[(summary["factor"] == factor) & (summary["strategy_name"] == f"{factor}-13{suffix}")].iloc[0]
            rows.append([factor, rule, fmt_pct(row["accuracy"]), fmt_num(row["final_wealth"]), fmt_pct(row["CAGR"]), fmt_ratio(row["sharpe"]), fmt_pct(row["max_drawdown"]), str(int(row["number_of_switches"]))])
    return {
        "caption": "Daily Winners and Losers comparison at k=13",
        "label": "tab:winners_losers_comparison",
        "data": pd.DataFrame(rows, columns=["Factor", "Rule", "Accuracy", "Final", "CAGR", "Sharpe", "MDD", "Switches"]),
    }


def row_from_summary(summary, frequency, factor, strategy_name):
    row = summary[summary["strategy_name"] == strategy_name].iloc[0]
    return [frequency, factor, strategy_name, fmt_num(row["final_wealth"]), fmt_pct(row["CAGR"]), fmt_pct(row["annualized_volatility"]), fmt_ratio(row["sharpe"]), fmt_pct(row["max_drawdown"]), str(int(row["number_of_switches"]))]


def table_classification_results():
    daily_knn = require_csv("paper_knn_strategy_summary.csv")
    daily_logit = require_csv("paper_logistic_strategy_summary.csv")
    weekly_knn = require_csv("paper_weekly_knn_strategy_summary.csv")
    rows = [
        row_from_summary(daily_logit, "Daily", "HML", "HML-Logit"),
        row_from_summary(daily_knn, "Daily", "RMW", "RMW-KNN11"),
        row_from_summary(daily_knn, "Daily", "CMA", "CMA-KNN3"),
        row_from_summary(daily_knn, "Daily", "SMB", "SMB-KNN3"),
        row_from_summary(weekly_knn, "Weekly", "HML", "HML-KNN15"),
        row_from_summary(weekly_knn, "Weekly", "RMW", "RMW-KNN11"),
        row_from_summary(weekly_knn, "Weekly", "SMB", "SMB-KNN23"),
        row_from_summary(weekly_knn, "Weekly", "CMA", "CMA-KNN23"),
    ]
    return {
        "caption": "Selected classification strategy results",
        "label": "tab:classification_results",
        "data": pd.DataFrame(rows, columns=["Frequency", "Factor", "Strategy", "Final", "CAGR", "Vol.", "Sharpe", "MDD", "Switches"]),
    }


def table_cost_012():
    configs = [
        ("Daily", "HML-Logit", "paper_logistic_strategy_returns.csv"),
        ("Daily", "RMW-KNN11", "paper_knn_strategy_returns.csv"),
        ("Daily", "CMA-KNN3", "paper_knn_strategy_returns.csv"),
        ("Daily", "SMB-KNN3", "paper_knn_strategy_returns.csv"),
        ("Weekly", "HML-KNN15", "paper_weekly_knn_strategy_returns.csv"),
        ("Weekly", "RMW-KNN11", "paper_weekly_knn_strategy_returns.csv"),
        ("Weekly", "SMB-KNN23", "paper_weekly_knn_strategy_returns.csv"),
        ("Weekly", "CMA-KNN23", "paper_weekly_knn_strategy_returns.csv"),
    ]
    rows = []
    for frequency, strategy, file_name in configs:
        base_metrics, returns0 = metrics_for_strategy_return_file(file_name, strategy, 0, frequency)
        spy = spy_returns_for_dates(returns0.index, frequency)
        spy_final = performance_metrics(spy, 52 if frequency == "Weekly" else 252)["final"]
        cells = []
        for bps in [0, 1, 2]:
            metrics, _ = metrics_for_strategy_return_file(file_name, strategy, bps, frequency)
            dagger = "$^{\\dagger}$" if metrics["final"] > spy_final else ""
            cells.append(f"{fmt_num(metrics['final'])}{dagger} / {fmt_ratio(metrics['sharpe'], 2)}")
        rows.append([frequency, strategy, fmt_num(spy_final), *cells])
    return {
        "caption": "Selected strategy performance at 0, 1, and 2 bps per switch",
        "label": "tab:cost_012",
        "data": pd.DataFrame(rows, columns=["Frequency", "Strategy", "SPY final", "0 bps", "1 bps", "2 bps"]),
        "notes": "Each strategy cell reports terminal wealth followed by annualized Sharpe ratio. Dagger indicates terminal wealth exceeds SPY over the same sample.",
    }


def table_oos():
    data = require_csv("paper_oos_strategy_summary.csv")
    rows = []
    for factor in ["CMA", "HML", "RMW", "SMB"]:
        row = data[data["factor"] == factor].iloc[0]
        rows.append([factor, str(int(row["observations"])), fmt_num(row["final_wealth"]), fmt_pct(row["CAGR"]), fmt_pct(row["annualized_volatility"]), fmt_ratio(row["sharpe"]), fmt_pct(row["max_drawdown"]), str(int(row["number_of_switches"])), fmt_pct(row["accuracy"])])
    return {
        "caption": "Out of sample strategy performance",
        "label": "tab:oos_results",
        "data": pd.DataFrame(rows, columns=["Factor", "Obs.", "Final", "CAGR", "Vol.", "Sharpe", "MDD", "Switches", "Acc."]),
    }


def table_single_proxy_regressions():
    data = require_csv("paper_proxy_validation_regressions.csv")
    rows = []
    for factor in ["SMB", "HML", "RMW", "CMA"]:
        group = data[data["factor"] == factor]
        for _, row in group.iterrows():
            term = "Intercept" if row["term"] == "alpha" else f"{factor} proxy"
            rows.append([factor, term, fmt_ratio(row["coefficient"], 5), fmt_ratio(row["standard_error"], 5), fmt_ratio(row["t_stat"], 2), fmt_pct(row["r_squared"]), str(int(row["observations"])) if term == "Intercept" else ""])
    return {
        "caption": "Single proxy regression estimates",
        "label": "tab:single_proxy_regressions",
        "data": pd.DataFrame(rows, columns=["Factor", "Term", "Coef.", "Std. Err.", "t-Stat", "R2", "Obs."]),
        "notes": "The dependent variable is the daily Fama French factor return. Conventional OLS standard errors.",
    }


def table_cost_sensitivity():
    configs = [
        ("Daily CMA Losers", "Daily", "CMA-13L", "paper_strategy_returns.csv"),
        ("Daily HML Logit", "Daily", "HML-Logit", "paper_logistic_strategy_returns.csv"),
        ("Weekly HML KNN", "Weekly", "HML-KNN15", "paper_weekly_knn_strategy_returns.csv"),
        ("Weekly SMB KNN", "Weekly", "SMB-KNN23", "paper_weekly_knn_strategy_returns.csv"),
    ]
    rows = []
    for label, frequency, strategy, file_name in configs:
        cells = []
        for bps in [0, 2, 5, 10]:
            metrics, _ = metrics_for_strategy_return_file(file_name, strategy, bps, frequency)
            cells.append(f"{fmt_num(metrics['final'])} / {fmt_ratio(metrics['sharpe'], 3)}")
        rows.append([label, *cells])
    return {
        "caption": "Transaction cost sensitivity: final wealth and Sharpe ratio",
        "label": "tab:cost_sensitivity",
        "data": pd.DataFrame(rows, columns=["Strategy", "0 bps", "2 bps", "5 bps", "10 bps"]),
    }


def table_weekly_knn_best():
    data = require_csv("paper_weekly_knn_strategy_summary.csv")
    rows = []
    for factor in ["CMA", "HML", "RMW", "SMB"]:
        row = data[data["factor"] == factor].sort_values(["sharpe", "final_wealth"], ascending=False).iloc[0]
        rows.append([factor, row["strategy_name"], FACTOR_LEGS[factor], fmt_pct(row["accuracy"]), fmt_num(row["final_wealth"]), fmt_pct(row["CAGR"]), fmt_pct(row["annualized_volatility"]), fmt_ratio(row["sharpe"]), fmt_pct(row["max_drawdown"]), str(int(row["number_of_switches"])), fmt_pct(row["spy_alpha_annualized"]), fmt_ratio(row["spy_beta"])])
    return {
        "caption": "Weekly KNN strategy performance, best k per factor",
        "label": "tab:weekly_knn_best_k",
        "data": pd.DataFrame(rows, columns=["Factor", "Strategy", "A/B", "Acc.", "Final", "CAGR", "Vol.", "Sharpe", "MDD", "Switches", "Alpha", "Beta"]),
        "notes": "Best k is selected by full sample Sharpe ratio at 2 bps per switch.",
    }


def table_logistic_daily_weekly():
    rows = []
    for frequency, file_name in [("Daily", "paper_logistic_strategy_summary.csv"), ("Weekly", "paper_weekly_logistic_strategy_summary.csv")]:
        data = require_csv(file_name)
        for factor in ["CMA", "HML", "RMW", "SMB"]:
            row = data[data["factor"] == factor].iloc[0]
            rows.append([frequency, factor, FACTOR_LEGS[factor], fmt_pct(row["accuracy"]), fmt_num(row["final_wealth"]), fmt_pct(row["CAGR"]), fmt_pct(row["annualized_volatility"]), fmt_ratio(row["sharpe"]), fmt_pct(row["max_drawdown"]), str(int(row["number_of_switches"])), f"{fmt_pct(row['spy_alpha_annualized'])} / {fmt_ratio(row['spy_beta'])}"])
    return {
        "caption": "Daily and weekly logistic regression strategy performance",
        "label": "tab:logistic_daily_weekly",
        "data": pd.DataFrame(rows, columns=["Freq.", "Factor", "A/B", "Acc.", "Final", "CAGR", "Vol.", "Sharpe", "MDD", "Switches", "Alpha / Beta"]),
    }


def table_knn_sensitivity():
    rows = []
    k_values = list(range(3, 26, 2))
    for frequency, file_name in [("Daily", "paper_knn_strategy_summary.csv"), ("Weekly", "paper_weekly_knn_strategy_summary.csv")]:
        data = require_csv(file_name)
        for factor in ["CMA", "HML", "RMW", "SMB"]:
            group = data[data["factor"] == factor]
            cells = []
            for k in k_values:
                row = group[group["k"] == k]
                cells.append(fmt_ratio(row["sharpe"].iloc[0], 2) if not row.empty else "")
            rows.append([frequency, factor, *cells])
    return {
        "caption": "Sensitivity of KNN to the number of neighbors k: Sharpe ratio",
        "label": "tab:knn_k_sensitivity_sharpe",
        "data": pd.DataFrame(rows, columns=["Freq.", "Factor", *[str(k) for k in k_values]]),
        "notes": "Values are annualized Sharpe ratios net of 2 bps per switch.",
    }


def table_payoff_asymmetry():
    data = require_csv("paper_payoff_asymmetry_table.csv")
    rows = []
    for _, row in data.iterrows():
        rows.append([row["Strategy"], fmt_pct(row["Accuracy"]), fmt_pct(row["mu_plus"], 3), fmt_pct(row["mu_minus"], 3), fmt_pct(row["p_star"]), fmt_ratio(row["Timing_bps_per_period"], 2), f"{fmt_ratio(row['PT_stat'], 2)} ({float(row['PT_p_one_sided']):.3f})"])
    return {
        "caption": "Payoff asymmetry and directional accuracy in selected strategies",
        "label": "tab:payoff_asymmetry",
        "data": pd.DataFrame(rows, columns=["Strategy", "Accuracy", "mu+", "mu-", "p*", "Timing", "PT Stat. (p)"]),
        "notes": "mu+ and mu- are average absolute ETF proxy spreads in correctly and incorrectly classified periods.",
    }


def build_tables():
    return [
        table_proxy_horse_race(),
        table_proxy_validation(),
        table_primary_strategy(),
        table_winners_losers(),
        table_classification_results(),
        table_cost_012(),
        table_oos(),
        table_single_proxy_regressions(),
        table_cost_sensitivity(),
        table_weekly_knn_best(),
        table_logistic_daily_weekly(),
        table_knn_sensitivity(),
        table_payoff_asymmetry(),
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    tex_path = write_tex_document(tables)
    pdf_path = write_pdf(tables)
    print(f"Saved LaTeX table appendix: {tex_path}")
    print(f"Saved PDF table appendix: {pdf_path}")


if __name__ == "__main__":
    main()
