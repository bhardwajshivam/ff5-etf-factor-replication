import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PREFIX = "paper"

FF5_PATH = DATA_DIR / "ff5_cleaned_daily.csv"
ALT_ETF_RETURNS_PATH = DATA_DIR / "alt_etf_daily_returns.csv"
BASE_ETF_RETURNS_PATH = DATA_DIR / "etf_daily_returns.csv"

COMMON_SAMPLE_START = pd.Timestamp("2014-08-01")
ROLLING_WINDOW_DAYS = 252 * 3
MIN_YEARLY_OBS = 80
INITIAL_WEALTH = 100.0
PRIMARY_COST_BPS = 2
PRIMARY_K = 13
K_VALUES = [3, 5, 7, 9, 11, 13, 15]
STRATEGY_TYPES = ["Winners", "Losers"]
OOS_TRAIN_YEARS = 3
DAILY_PERIODS_PER_YEAR = 252
WEEKLY_PERIODS_PER_YEAR = 52
KNN_NEIGHBORS = list(range(3, 26, 2))
DAILY_KNN_WINDOWS = [3, 5, 10, 21]
WEEKLY_KNN_WINDOWS = [1, 2, 4, 8]
DAILY_KNN_MIN_TRAINING_OBS = 252
WEEKLY_KNN_MIN_TRAINING_OBS = 52
LOGIT_L2_PENALTY = 1.0
LOGIT_MAX_ITER = 25

PROXY_DEFINITIONS = {
    "SMB": {
        "long": {"IJR": 1.0},
        "short": {"SPY": 1.0},
        "description": "Small-cap minus large-cap ETF proxy",
    },
    "HML": {
        "long": {"SPYV": 1.0},
        "short": {"SPYG": 1.0},
        "description": "Value minus growth ETF proxy",
    },
    "RMW": {
        "long": {"QUAL": 1.0},
        "short": {"ARKK": 1.0},
        "description": "Quality/profitability proxy minus speculative innovation proxy",
    },
    "CMA": {
        "long": {"SPHQ": 1.0},
        "short": {"QQQ": 1.0},
        "description": "Quality/conservative proxy minus Nasdaq/aggressive growth proxy",
    },
}


def load_inputs():
    ff5 = pd.read_csv(FF5_PATH, parse_dates=["date"]).set_index("date").sort_index()
    alt = pd.read_csv(ALT_ETF_RETURNS_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    base = pd.read_csv(BASE_ETF_RETURNS_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    alt.index.name = "date"
    base.index.name = "date"
    etf_returns = alt.combine_first(base).apply(pd.to_numeric, errors="coerce")
    return ff5[["MKT", "SMB", "HML", "RMW", "CMA", "RF"]], etf_returns


def weighted_return(etf_returns, weights):
    output = pd.Series(0.0, index=etf_returns.index)
    for ticker, weight in weights.items():
        if ticker not in etf_returns.columns:
            raise ValueError(f"Missing required ETF return column: {ticker}")
        output = output.add(etf_returns[ticker] * weight, fill_value=np.nan)
    return output


def build_proxy_panel(ff5, etf_returns):
    rows = []
    proxy_returns = pd.DataFrame(index=etf_returns.index)
    leg_returns = {}

    for factor, spec in PROXY_DEFINITIONS.items():
        long_return = weighted_return(etf_returns, spec["long"])
        short_return = weighted_return(etf_returns, spec["short"])
        proxy_return = long_return - short_return
        proxy_returns[f"{factor}_proxy"] = proxy_return
        leg_returns[factor] = (long_return, short_return)

    required = [*PROXY_DEFINITIONS.keys(), *proxy_returns.columns]
    validation_data = ff5[list(PROXY_DEFINITIONS)].join(proxy_returns, how="inner")
    validation_data = validation_data[validation_data.index >= COMMON_SAMPLE_START].dropna(subset=required)
    common_index = validation_data.index

    for factor, spec in PROXY_DEFINITIONS.items():
        long_return, short_return = leg_returns[factor]
        panel = pd.DataFrame(
            {
                "date": common_index,
                "factor": factor,
                "A_return": long_return.reindex(common_index),
                "B_return": short_return.reindex(common_index),
                "proxy_return": proxy_returns[f"{factor}_proxy"].reindex(common_index),
                "official_factor_return": ff5[factor].reindex(common_index),
                "A_members": ", ".join(spec["long"].keys()),
                "B_members": ", ".join(spec["short"].keys()),
            }
        ).dropna()
        panel["true_label"] = np.where(panel["A_return"] >= panel["B_return"], 1, -1)
        rows.append(panel)

    strategy_panel = pd.concat(rows, ignore_index=True).sort_values(["factor", "date"])
    return validation_data, strategy_panel


def compound_return(series):
    series = series.dropna()
    return (1.0 + series).prod() - 1.0 if len(series) else np.nan


def build_weekly_strategy_panel(strategy_panel):
    rows = []
    return_columns = ["A_return", "B_return", "proxy_return", "official_factor_return"]
    for factor, group in strategy_panel.groupby("factor"):
        group = group.sort_values("date").set_index("date")
        weekly_returns = group[return_columns].resample("W-FRI").apply(compound_return)
        weekly_returns["factor"] = factor
        weekly_returns["A_members"] = group["A_members"].resample("W-FRI").last()
        weekly_returns["B_members"] = group["B_members"].resample("W-FRI").last()
        weekly_returns = weekly_returns.dropna(subset=["A_return", "B_return"])
        weekly_returns["true_label"] = np.where(weekly_returns["A_return"] >= weekly_returns["B_return"], 1, -1)
        rows.append(weekly_returns.reset_index())
    return pd.concat(rows, ignore_index=True).sort_values(["factor", "date"])


def build_weekly_spy_returns(etf_returns, weekly_dates):
    spy = etf_returns["SPY"].dropna().resample("W-FRI").apply(compound_return)
    return pd.DataFrame({"SPY": spy.reindex(pd.DatetimeIndex(sorted(weekly_dates)))})


def ols(y, x):
    frame = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(frame) <= x.shape[1] + 1:
        return None
    y_values = frame["y"].to_numpy(dtype=float)
    x_values = frame.drop(columns="y").to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(frame)), x_values])
    coefficients = np.linalg.lstsq(x_design, y_values, rcond=None)[0]
    fitted = x_design @ coefficients
    residual = y_values - fitted
    dof = max(len(frame) - x_design.shape[1], 1)
    sigma2 = float((residual @ residual) / dof)
    standard_errors = np.sqrt(np.diag(sigma2 * np.linalg.pinv(x_design.T @ x_design)))
    ss_resid = float((residual ** 2).sum())
    ss_total = float(((y_values - y_values.mean()) ** 2).sum())
    names = ["alpha", *x.columns]
    return {
        "observations": len(frame),
        "coefficients": pd.Series(coefficients, index=names),
        "standard_errors": pd.Series(standard_errors, index=names),
        "tstats": pd.Series(coefficients / np.where(standard_errors == 0, np.nan, standard_errors), index=names),
        "r_squared": 1.0 - ss_resid / ss_total if ss_total else np.nan,
        "fit": pd.DataFrame({"actual": y_values, "fitted": fitted, "residual": residual}, index=frame.index),
    }


def proxy_validation_tables(validation_data):
    summary_rows = []
    regression_rows = []
    fit_rows = []
    for factor in PROXY_DEFINITIONS:
        proxy_col = f"{factor}_proxy"
        result = ols(validation_data[factor], validation_data[[proxy_col]])
        if result is None:
            continue
        pair = validation_data[[factor, proxy_col]].dropna()
        raw_error = pair[factor] - pair[proxy_col]
        residual = result["fit"]["residual"]
        beta = result["coefficients"][proxy_col]
        summary_rows.append(
            {
                "factor": factor,
                "proxy_definition": proxy_formula_text(factor),
                "description": PROXY_DEFINITIONS[factor]["description"],
                "date_start": pair.index.min().date(),
                "date_end": pair.index.max().date(),
                "observations": len(pair),
                "correlation": pair[factor].corr(pair[proxy_col]),
                "sign_match_rate": (np.sign(pair[factor]) == np.sign(pair[proxy_col])).mean(),
                "alpha_daily": result["coefficients"]["alpha"],
                "alpha_annualized": result["coefficients"]["alpha"] * 252,
                "beta": beta,
                "beta_tstat": result["tstats"][proxy_col],
                "r_squared": result["r_squared"],
                "raw_tracking_error": raw_error.std() * np.sqrt(252),
                "residual_volatility": residual.std() * np.sqrt(252),
            }
        )
        for term in ["alpha", proxy_col]:
            regression_rows.append(
                {
                    "factor": factor,
                    "term": term,
                    "coefficient": result["coefficients"][term],
                    "standard_error": result["standard_errors"][term],
                    "t_stat": result["tstats"][term],
                    "r_squared": result["r_squared"],
                    "observations": result["observations"],
                }
            )
        fit = result["fit"].copy()
        fit["factor"] = factor
        fit["proxy_return"] = pair[proxy_col].reindex(fit.index)
        fit_rows.append(fit.reset_index().rename(columns={"index": "date"}))
    return pd.DataFrame(summary_rows), pd.DataFrame(regression_rows), pd.concat(fit_rows, ignore_index=True)


def yearly_proxy_validation(validation_data):
    rows = []
    for factor in PROXY_DEFINITIONS:
        proxy_col = f"{factor}_proxy"
        pair = validation_data[[factor, proxy_col]].dropna()
        for year, group in pair.groupby(pair.index.year):
            if len(group) < MIN_YEARLY_OBS:
                continue
            result = ols(group[factor], group[[proxy_col]])
            rows.append(
                {
                    "factor": factor,
                    "year": year,
                    "observations": len(group),
                    "correlation": group[factor].corr(group[proxy_col]),
                    "r_squared": result["r_squared"],
                    "beta": result["coefficients"][proxy_col],
                    "raw_tracking_error": (group[factor] - group[proxy_col]).std() * np.sqrt(252),
                    "official_annual_return": (1.0 + group[factor]).prod() - 1.0,
                    "proxy_annual_return": (1.0 + group[proxy_col]).prod() - 1.0,
                }
            )
    return pd.DataFrame(rows)


def rolling_proxy_validation(validation_data):
    rows = []
    for factor in PROXY_DEFINITIONS:
        proxy_col = f"{factor}_proxy"
        pair = validation_data[[factor, proxy_col]].dropna()
        for end_position in range(ROLLING_WINDOW_DAYS, len(pair) + 1):
            window = pair.iloc[end_position - ROLLING_WINDOW_DAYS:end_position]
            result = ols(window[factor], window[[proxy_col]])
            rows.append(
                {
                    "factor": factor,
                    "date": window.index[-1],
                    "rolling_correlation": window[factor].corr(window[proxy_col]),
                    "rolling_r_squared": result["r_squared"],
                    "rolling_beta": result["coefficients"][proxy_col],
                    "rolling_tracking_error": (window[factor] - window[proxy_col]).std() * np.sqrt(252),
                }
            )
    return pd.DataFrame(rows)


def proxy_formula_text(factor):
    spec = PROXY_DEFINITIONS[factor]
    long_side = " + ".join(format_weighted_ticker(t, w) for t, w in spec["long"].items())
    short_side = " + ".join(format_weighted_ticker(t, w) for t, w in spec["short"].items())
    return f"{long_side} - {short_side}"


def format_weighted_ticker(ticker, weight):
    return ticker if np.isclose(weight, 1.0) else f"{weight:g}*{ticker}"


def strategy_predictions(strategy_panel, k, strategy_type):
    rows = []
    for factor, factor_panel in strategy_panel.groupby("factor"):
        factor_panel = factor_panel.sort_values("date").copy()
        rolling_sum = factor_panel["true_label"].shift(1).rolling(k).sum()
        predicted = np.where(rolling_sum >= 0, 1, -1)
        if strategy_type == "Losers":
            predicted = -predicted
        output = factor_panel.loc[rolling_sum.notna()].copy()
        output["k"] = k
        output["strategy_type"] = strategy_type
        output["strategy_name"] = f"{factor}-{k}{strategy_type[0]}"
        output["predicted_label"] = predicted[rolling_sum.notna()]
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def apply_costs(predictions, cost_bps):
    output = predictions.sort_values(["factor", "strategy_name", "date"]).copy()
    output["chosen_return_before_cost"] = np.where(output["predicted_label"] == 1, output["A_return"], output["B_return"])
    output["correct"] = (output["predicted_label"] == output["true_label"]).astype(int)
    output["position_switch"] = (
        output.groupby(["factor", "strategy_name"])["predicted_label"].diff().fillna(0).ne(0).astype(int)
    )
    output["transaction_cost_bps"] = cost_bps
    output["transaction_cost"] = output["position_switch"] * cost_bps / 10000.0
    output["strategy_return"] = output["chosen_return_before_cost"] - output["transaction_cost"]
    return output


def max_drawdown(returns):
    wealth = (1.0 + returns.dropna()).cumprod()
    if wealth.empty:
        return np.nan
    return (wealth / wealth.cummax() - 1.0).min()


def financial_metrics(returns, spy_returns=None, periods_per_year=DAILY_PERIODS_PER_YEAR):
    returns = returns.dropna()
    if returns.empty:
        return {}
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / periods_per_year
    total_return = wealth.iloc[-1] - 1.0
    annualized_return = returns.mean() * periods_per_year
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    downside = returns[returns < 0]
    downside_volatility = downside.std() * np.sqrt(periods_per_year) if len(downside) else np.nan
    metrics = {
        "observations": len(returns),
        "final_wealth": INITIAL_WEALTH * wealth.iloc[-1],
        "cumulative_return": total_return,
        "CAGR": (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and total_return > -1 else np.nan,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": annualized_return / annualized_volatility if annualized_volatility else np.nan,
        "sortino": annualized_return / downside_volatility if downside_volatility else np.nan,
        "max_drawdown": max_drawdown(returns),
    }
    if spy_returns is not None:
        aligned = pd.concat([returns.rename("strategy"), spy_returns.rename("SPY")], axis=1).dropna()
        if len(aligned) > 5:
            result = ols(aligned["strategy"], aligned[["SPY"]])
            metrics["spy_alpha_annualized"] = result["coefficients"]["alpha"] * periods_per_year
            metrics["spy_beta"] = result["coefficients"]["SPY"]
            active = aligned["strategy"] - aligned["SPY"]
            active_volatility = active.std() * np.sqrt(periods_per_year)
            metrics["information_ratio_vs_spy"] = (
                active.mean() * periods_per_year / active_volatility if active_volatility else np.nan
            )
    return metrics


def paper_return_efficiency(group, return_column):
    r_a = group["A_return"]
    r_b = group["B_return"]
    r_strategy = group[return_column]
    perfect = np.maximum(r_a, r_b)
    worst = np.minimum(r_a, r_b)
    denominator = perfect.sum() - worst.sum()
    return (r_strategy.sum() - worst.sum()) / denominator if denominator else np.nan


def summarize_strategy_returns(strategy_returns, etf_returns, periods_per_year=DAILY_PERIODS_PER_YEAR):
    rows = []
    spy = etf_returns["SPY"]
    for keys, group in strategy_returns.groupby(["factor", "strategy_name", "strategy_type", "k", "transaction_cost_bps"]):
        factor, strategy_name, strategy_type, k, cost_bps = keys
        group = group.sort_values("date")
        returns = pd.Series(group["strategy_return"].to_numpy(), index=group["date"])
        spy_returns = spy.reindex(returns.index)
        rows.append(
            {
                "factor": factor,
                "strategy_name": strategy_name,
                "strategy_type": strategy_type,
                "k": k,
                "transaction_cost_bps": cost_bps,
                "accuracy": group["correct"].mean(),
                "number_of_switches": int(group["position_switch"].sum()),
                "turnover": group["position_switch"].mean(),
                "REI": paper_return_efficiency(group, "strategy_return"),
                "A_members": group["A_members"].iloc[-1],
                "B_members": group["B_members"].iloc[-1],
                **financial_metrics(returns, spy_returns, periods_per_year=periods_per_year),
            }
        )
    return pd.DataFrame(rows)


def build_all_strategy_returns(strategy_panel):
    predictions = pd.concat(
        [strategy_predictions(strategy_panel, k, strategy_type) for k in K_VALUES for strategy_type in STRATEGY_TYPES],
        ignore_index=True,
    )
    return apply_costs(predictions, PRIMARY_COST_BPS)


def rolling_compounded(series, window):
    return (1.0 + series.shift(1)).rolling(window).apply(np.prod, raw=True) - 1.0


def rolling_volatility(series, window, periods_per_year):
    return series.shift(1).rolling(window).std(ddof=0) * np.sqrt(periods_per_year)


def build_knn_feature_panel(strategy_panel, windows, periods_per_year):
    rows = []
    for factor, group in strategy_panel.groupby("factor"):
        group = group.sort_values("date").copy()
        features = group[["date", "factor", "A_return", "B_return", "proxy_return", "A_members", "B_members", "true_label"]].copy()
        for window in windows:
            features[f"A_return_lag_{window}"] = rolling_compounded(group["A_return"], window)
            features[f"B_return_lag_{window}"] = rolling_compounded(group["B_return"], window)
            features[f"proxy_return_lag_{window}"] = rolling_compounded(group["proxy_return"], window)
            features[f"A_volatility_lag_{window}"] = rolling_volatility(group["A_return"], window, periods_per_year)
            features[f"B_volatility_lag_{window}"] = rolling_volatility(group["B_return"], window, periods_per_year)
            features[f"proxy_volatility_lag_{window}"] = rolling_volatility(group["proxy_return"], window, periods_per_year)
        rows.append(features)
    return pd.concat(rows, ignore_index=True).dropna().sort_values(["factor", "date"])


def knn_feature_columns(feature_panel):
    excluded = {"date", "factor", "A_return", "B_return", "proxy_return", "A_members", "B_members", "true_label"}
    return [column for column in feature_panel.columns if column not in excluded]


def knn_ab_predictions(feature_panel, k_values, min_training_obs):
    feature_columns = knn_feature_columns(feature_panel)
    rows = []
    for factor, group in feature_panel.groupby("factor"):
        group = group.sort_values("date").reset_index(drop=True)
        x_all = group[feature_columns].to_numpy(dtype=float)
        y_all = group["true_label"].to_numpy(dtype=int)
        for k_neighbors in k_values:
            for i in range(min_training_obs, len(group)):
                x_train = x_all[:i]
                y_train = y_all[:i]
                x_current = x_all[i]
                means = np.nanmean(x_train, axis=0)
                stds = np.nanstd(x_train, axis=0)
                stds = np.where(stds == 0, 1.0, stds)
                x_train_scaled = (x_train - means) / stds
                x_current_scaled = (x_current - means) / stds
                distances = np.sqrt(np.sum((x_train_scaled - x_current_scaled) ** 2, axis=1))
                neighbor_idx = np.argsort(distances)[:k_neighbors]
                predicted_label = 1 if y_train[neighbor_idx].mean() >= 0 else -1
                row = group.iloc[i].to_dict()
                row["k"] = k_neighbors
                row["strategy_type"] = "KNN"
                row["strategy_name"] = f"{factor}-KNN{k_neighbors}"
                row["predicted_label"] = predicted_label
                row["features_used"] = ",".join(feature_columns)
                rows.append(row)
    return pd.DataFrame(rows)


def build_knn_strategy_returns(strategy_panel, windows, periods_per_year, min_training_obs):
    feature_panel = build_knn_feature_panel(strategy_panel, windows, periods_per_year)
    predictions = knn_ab_predictions(feature_panel, KNN_NEIGHBORS, min_training_obs)
    return feature_panel, apply_costs(predictions, PRIMARY_COST_BPS)


def fit_logistic_irls(x_train, y_train, beta_start=None):
    x_design = np.column_stack([np.ones(len(x_train)), x_train])
    y_binary = (y_train == 1).astype(float)
    beta = np.zeros(x_design.shape[1]) if beta_start is None else beta_start.copy()
    penalty = np.r_[0.0, np.repeat(LOGIT_L2_PENALTY, x_train.shape[1])]
    for _ in range(LOGIT_MAX_ITER):
        probability = expit(np.clip(x_design @ beta, -35, 35))
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = x_design.T @ (probability - y_binary) + penalty * beta
        hessian = (x_design.T * weights) @ x_design + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def logistic_ab_predictions(feature_panel, min_training_obs):
    feature_columns = knn_feature_columns(feature_panel)
    rows = []
    for factor, group in feature_panel.groupby("factor"):
        group = group.sort_values("date").reset_index(drop=True)
        x_all = group[feature_columns].to_numpy(dtype=float)
        y_all = group["true_label"].to_numpy(dtype=int)
        beta = None
        for i in range(min_training_obs, len(group)):
            x_train = x_all[:i]
            y_train = y_all[:i]
            means = np.nanmean(x_train, axis=0)
            stds = np.nanstd(x_train, axis=0)
            stds = np.where(stds == 0, 1.0, stds)
            x_train_scaled = (x_train - means) / stds
            x_current_scaled = (x_all[i] - means) / stds
            beta = fit_logistic_irls(x_train_scaled, y_train, beta)
            probability_a = expit(np.clip(np.r_[1.0, x_current_scaled] @ beta, -35, 35))
            row = group.iloc[i].to_dict()
            row["k"] = 0
            row["strategy_type"] = "Logistic"
            row["strategy_name"] = f"{factor}-Logit"
            row["predicted_label"] = 1 if probability_a >= 0.5 else -1
            row["predicted_probability_A"] = probability_a
            row["features_used"] = ",".join(feature_columns)
            rows.append(row)
    return pd.DataFrame(rows)


def build_logistic_strategy_returns(strategy_panel, windows, periods_per_year, min_training_obs):
    feature_panel = build_knn_feature_panel(strategy_panel, windows, periods_per_year)
    predictions = logistic_ab_predictions(feature_panel, min_training_obs)
    return feature_panel, apply_costs(predictions, PRIMARY_COST_BPS)


def plot_knn_strategy_vs_spy(summary, strategy_returns, spy_returns, output_name, title):
    top = (
        summary[summary["transaction_cost_bps"] == PRIMARY_COST_BPS]
        .sort_values("sharpe", ascending=False)
        .groupby("factor")
        .head(1)
    )
    plt.figure(figsize=(13, 7))
    date_range = []
    for _, row in top.iterrows():
        subset = strategy_returns[
            (strategy_returns["factor"] == row["factor"])
            & (strategy_returns["strategy_name"] == row["strategy_name"])
        ].sort_values("date")
        returns = pd.Series(subset["strategy_return"].to_numpy(), index=subset["date"])
        wealth = INITIAL_WEALTH * (1.0 + returns).cumprod()
        plt.plot(wealth.index, wealth, label=row["strategy_name"], linewidth=2)
        date_range.extend(wealth.index.tolist())
    if date_range:
        spy = spy_returns["SPY"].loc[min(date_range):max(date_range)].dropna()
        spy_wealth = INITIAL_WEALTH * (1.0 + spy).cumprod()
        plt.plot(spy_wealth.index, spy_wealth, label="SPY buy-and-hold", color="black", linewidth=3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close()


def plot_knn_k_sensitivity(summary, output_name, title):
    subset = summary[summary["transaction_cost_bps"] == PRIMARY_COST_BPS]
    pivot = subset.pivot_table(index="factor", columns="k", values="sharpe", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("KNN neighbors")
    fig.colorbar(image, ax=ax, label="Sharpe")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close()


def out_of_sample_selection(
    strategy_returns,
    etf_returns,
    periods_per_year=DAILY_PERIODS_PER_YEAR,
    min_obs_per_full_year=200,
):
    rows = []
    choices = []
    strategy_returns = strategy_returns.sort_values("date")
    for factor in PROXY_DEFINITIONS:
        factor_data = strategy_returns[strategy_returns["factor"] == factor]
        yearly_counts = factor_data.groupby(factor_data["date"].dt.year)["date"].nunique()
        full_years = set(yearly_counts[yearly_counts >= min_obs_per_full_year].index)
        years = sorted(yearly_counts.index)
        for test_year in years:
            train_years = set(range(test_year - OOS_TRAIN_YEARS, test_year))
            if not train_years.issubset(full_years):
                continue
            train_start = pd.Timestamp(f"{test_year - OOS_TRAIN_YEARS}-01-01")
            train_end = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year}-12-31")
            train = factor_data[(factor_data["date"] >= train_start) & (factor_data["date"] <= train_end)]
            if train.empty:
                continue
            train_summary = summarize_strategy_returns(train, etf_returns, periods_per_year=periods_per_year)
            chosen = train_summary.sort_values(["sharpe", "REI"], ascending=False).iloc[0]
            test = factor_data[
                (factor_data["date"] >= test_start)
                & (factor_data["date"] <= test_end)
                & (factor_data["strategy_name"] == chosen["strategy_name"])
            ].copy()
            if test.empty:
                continue
            test["selected_training_sharpe"] = chosen["sharpe"]
            test["selected_training_REI"] = chosen["REI"]
            rows.append(test)
            choices.append(
                {
                    "factor": factor,
                    "test_year": test_year,
                    "selected_strategy": chosen["strategy_name"],
                    "selected_k": chosen["k"],
                    "selected_strategy_type": chosen["strategy_type"],
                    "training_sharpe": chosen["sharpe"],
                    "training_REI": chosen["REI"],
                    "test_observations": len(test),
                }
            )
    oos_returns = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return oos_returns, pd.DataFrame(choices)


def benchmark_tables(strategy_panel, etf_returns):
    rows = []
    spy = etf_returns["SPY"].reindex(pd.DatetimeIndex(strategy_panel["date"].unique())).dropna()
    rows.append({"benchmark": "SPY buy-and-hold", **financial_metrics(spy)})
    for factor, group in strategy_panel.groupby("factor"):
        group = group.sort_values("date")
        for name, col in [("A buy-and-hold", "A_return"), ("B buy-and-hold", "B_return"), ("ETF proxy spread", "proxy_return")]:
            returns = pd.Series(group[col].to_numpy(), index=group["date"])
            rows.append({"benchmark": f"{factor} {name}", **financial_metrics(returns, etf_returns["SPY"].reindex(returns.index))})
    return pd.DataFrame(rows)


def plot_proxy_validation(summary):
    plt.figure(figsize=(12, 7))
    x = np.arange(len(summary))
    width = 0.25
    plt.bar(x - width, summary["r_squared"], width=width, label="R-squared")
    plt.bar(x, summary["correlation"].abs(), width=width, label="Absolute correlation")
    plt.bar(x + width, summary["sign_match_rate"], width=width, label="Sign match rate")
    plt.xticks(x, summary["factor"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("ETF Proxy Validation Scorecard, Common Sample")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_proxy_validation_scorecard.png", dpi=200)
    plt.close()


def plot_rolling_r2(rolling):
    plt.figure(figsize=(13, 7))
    for factor, group in rolling.groupby("factor"):
        plt.plot(group["date"], group["rolling_r_squared"], label=factor, linewidth=1.8)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% R2 reference")
    plt.ylim(0, 1)
    plt.title("Rolling 3-Year R-squared: Official Factors Explained by ETF Proxies")
    plt.xlabel("Date")
    plt.ylabel("Rolling R-squared")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_rolling_proxy_r2.png", dpi=200)
    plt.close()


def plot_strategy_vs_spy(summary, strategy_returns, etf_returns):
    selected = summary[
        (summary["transaction_cost_bps"] == PRIMARY_COST_BPS)
        & (summary["strategy_type"] == "Losers")
        & (summary["k"] == PRIMARY_K)
    ]
    plt.figure(figsize=(13, 7))
    date_range = []
    for _, row in selected.iterrows():
        subset = strategy_returns[
            (strategy_returns["factor"] == row["factor"])
            & (strategy_returns["strategy_name"] == row["strategy_name"])
        ].sort_values("date")
        returns = pd.Series(subset["strategy_return"].to_numpy(), index=subset["date"])
        wealth = INITIAL_WEALTH * (1.0 + returns).cumprod()
        plt.plot(wealth.index, wealth, label=row["strategy_name"], linewidth=2)
        date_range.extend(wealth.index.tolist())
    if date_range:
        spy = etf_returns["SPY"].loc[min(date_range):max(date_range)].dropna()
        spy_wealth = INITIAL_WEALTH * (1.0 + spy).cumprod()
        plt.plot(spy_wealth.index, spy_wealth, label="SPY buy-and-hold", color="black", linewidth=3)
    plt.title(f"Primary Strategy Wealth vs SPY, k={PRIMARY_K}, Cost={PRIMARY_COST_BPS} bps")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_primary_strategy_vs_spy.png", dpi=200)
    plt.close()


def plot_k_sensitivity(summary):
    subset = summary[summary["transaction_cost_bps"] == PRIMARY_COST_BPS]
    pivot = subset.pivot_table(index="factor", columns="k", values="sharpe", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Best Sharpe by Factor and k, Cost={PRIMARY_COST_BPS} bps")
    ax.set_xlabel("k")
    fig.colorbar(image, ax=ax, label="Sharpe")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_k_sensitivity_heatmap.png", dpi=200)
    plt.close()


def plot_weekly_strategy_vs_spy(summary, strategy_returns, weekly_spy_returns):
    selected = summary[
        (summary["transaction_cost_bps"] == PRIMARY_COST_BPS)
        & (summary["strategy_type"] == "Losers")
        & (summary["k"] == PRIMARY_K)
    ]
    plt.figure(figsize=(13, 7))
    date_range = []
    for _, row in selected.iterrows():
        subset = strategy_returns[
            (strategy_returns["factor"] == row["factor"])
            & (strategy_returns["strategy_name"] == row["strategy_name"])
        ].sort_values("date")
        returns = pd.Series(subset["strategy_return"].to_numpy(), index=subset["date"])
        wealth = INITIAL_WEALTH * (1.0 + returns).cumprod()
        plt.plot(wealth.index, wealth, label=row["strategy_name"], linewidth=2)
        date_range.extend(wealth.index.tolist())
    if date_range:
        spy = weekly_spy_returns["SPY"].loc[min(date_range):max(date_range)].dropna()
        spy_wealth = INITIAL_WEALTH * (1.0 + spy).cumprod()
        plt.plot(spy_wealth.index, spy_wealth, label="SPY buy-and-hold", color="black", linewidth=3)
    plt.title(f"Weekly Primary Strategy Wealth vs SPY, k={PRIMARY_K}, Cost={PRIMARY_COST_BPS} bps")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_primary_strategy_vs_spy.png", dpi=200)
    plt.close()


def plot_weekly_k_sensitivity(summary):
    subset = summary[summary["transaction_cost_bps"] == PRIMARY_COST_BPS]
    pivot = subset.pivot_table(index="factor", columns="k", values="sharpe", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Weekly Best Sharpe by Factor and k, Cost={PRIMARY_COST_BPS} bps")
    ax.set_xlabel("k")
    fig.colorbar(image, ax=ax, label="Sharpe")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_k_sensitivity_heatmap.png", dpi=200)
    plt.close()


def write_method_note(validation_data):
    lines = [
        "Publication analysis method note",
        "",
        f"Common sample: {validation_data.index.min().date()} to {validation_data.index.max().date()}",
        f"Primary trading cost: {PRIMARY_COST_BPS} bps per switch",
        f"Primary baseline strategy: Losers, k={PRIMARY_K}",
        f"k sensitivity: {K_VALUES}",
        "",
        "ETF proxy definitions:",
    ]
    for factor in PROXY_DEFINITIONS:
        lines.append(f"  {factor}: {proxy_formula_text(factor)}; {PROXY_DEFINITIONS[factor]['description']}")
    lines.extend(
        [
            "",
            "Proxy validation regression:",
            "  Official FF factor return = alpha + beta * ETF proxy spread return + error",
            "",
            "Strategy rule:",
            "  A daily label is +1 if proxy side A beats side B and -1 otherwise.",
            "  Winners follows the majority label from the previous k trading days.",
            "  Losers takes the opposite of the previous-k-day majority label.",
            "  A separate weekly version compounds A and B leg returns to Friday weeks and applies the same rule to weekly labels.",
            "",
            "KNN strategy:",
            f"  KNN uses expanding-window classification with neighbor counts {KNN_NEIGHBORS}.",
            f"  Daily KNN features use lag windows {DAILY_KNN_WINDOWS}; weekly KNN features use lag windows {WEEKLY_KNN_WINDOWS}.",
            f"  Features include trailing A-leg return, B-leg return, proxy-spread return, and their corresponding volatilities.",
            f"  KNN returns are also reported net of {PRIMARY_COST_BPS} bps per position switch.",
            "",
            "Logistic regression strategy:",
            "  Logistic regression uses the same expanding-window classification setup as KNN.",
            "  The fitted probability is the probability that ETF leg A beats ETF leg B in the next trading period.",
            "  The strategy holds A when the predicted probability is at least 50 percent and B otherwise.",
            f"  Logistic returns are reported net of {PRIMARY_COST_BPS} bps per position switch.",
            "",
            "Out-of-sample selection:",
            f"  For each factor-year, select the best k/strategy type using the previous {OOS_TRAIN_YEARS} years by Sharpe ratio, then evaluate during the next calendar year.",
        ]
    )
    (OUTPUT_DIR / f"{OUTPUT_PREFIX}_method_note.txt").write_text("\n".join(lines), encoding="utf-8")


def run_publication_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ff5, etf_returns = load_inputs()
    validation_data, strategy_panel = build_proxy_panel(ff5, etf_returns)
    etf_returns = etf_returns.reindex(validation_data.index)

    validation_summary, single_regressions, validation_fit = proxy_validation_tables(validation_data)
    yearly_validation = yearly_proxy_validation(validation_data)
    rolling_validation = rolling_proxy_validation(validation_data)

    strategy_returns = build_all_strategy_returns(strategy_panel)
    strategy_summary = summarize_strategy_returns(strategy_returns, etf_returns)
    knn_feature_panel, knn_strategy_returns = build_knn_strategy_returns(
        strategy_panel,
        DAILY_KNN_WINDOWS,
        DAILY_PERIODS_PER_YEAR,
        DAILY_KNN_MIN_TRAINING_OBS,
    )
    knn_strategy_summary = summarize_strategy_returns(knn_strategy_returns, etf_returns)
    logistic_feature_panel, logistic_strategy_returns = build_logistic_strategy_returns(
        strategy_panel,
        DAILY_KNN_WINDOWS,
        DAILY_PERIODS_PER_YEAR,
        DAILY_KNN_MIN_TRAINING_OBS,
    )
    logistic_strategy_summary = summarize_strategy_returns(logistic_strategy_returns, etf_returns)
    benchmark_summary = benchmark_tables(strategy_panel, etf_returns)
    oos_returns, oos_choices = out_of_sample_selection(strategy_returns, etf_returns)
    oos_summary = summarize_strategy_returns(oos_returns, etf_returns) if not oos_returns.empty else pd.DataFrame()

    weekly_strategy_panel = build_weekly_strategy_panel(strategy_panel)
    weekly_spy_returns = build_weekly_spy_returns(etf_returns, weekly_strategy_panel["date"].unique())
    weekly_strategy_returns = build_all_strategy_returns(weekly_strategy_panel)
    weekly_strategy_summary = summarize_strategy_returns(
        weekly_strategy_returns,
        weekly_spy_returns,
        periods_per_year=WEEKLY_PERIODS_PER_YEAR,
    )
    weekly_knn_feature_panel, weekly_knn_strategy_returns = build_knn_strategy_returns(
        weekly_strategy_panel,
        WEEKLY_KNN_WINDOWS,
        WEEKLY_PERIODS_PER_YEAR,
        WEEKLY_KNN_MIN_TRAINING_OBS,
    )
    weekly_knn_strategy_summary = summarize_strategy_returns(
        weekly_knn_strategy_returns,
        weekly_spy_returns,
        periods_per_year=WEEKLY_PERIODS_PER_YEAR,
    )
    weekly_logistic_feature_panel, weekly_logistic_strategy_returns = build_logistic_strategy_returns(
        weekly_strategy_panel,
        WEEKLY_KNN_WINDOWS,
        WEEKLY_PERIODS_PER_YEAR,
        WEEKLY_KNN_MIN_TRAINING_OBS,
    )
    weekly_logistic_strategy_summary = summarize_strategy_returns(
        weekly_logistic_strategy_returns,
        weekly_spy_returns,
        periods_per_year=WEEKLY_PERIODS_PER_YEAR,
    )
    weekly_oos_returns, weekly_oos_choices = out_of_sample_selection(
        weekly_strategy_returns,
        weekly_spy_returns,
        periods_per_year=WEEKLY_PERIODS_PER_YEAR,
        min_obs_per_full_year=40,
    )
    weekly_oos_summary = (
        summarize_strategy_returns(
            weekly_oos_returns,
            weekly_spy_returns,
            periods_per_year=WEEKLY_PERIODS_PER_YEAR,
        )
        if not weekly_oos_returns.empty
        else pd.DataFrame()
    )

    validation_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_proxy_validation_summary.csv", index=False)
    single_regressions.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_proxy_validation_regressions.csv", index=False)
    validation_fit.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_proxy_validation_actual_fitted.csv", index=False)
    yearly_validation.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_proxy_validation_yearly.csv", index=False)
    rolling_validation.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_proxy_validation_rolling_3y.csv", index=False)
    strategy_panel.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_strategy_panel.csv", index=False)
    strategy_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_strategy_returns.csv", index=False)
    strategy_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_strategy_summary.csv", index=False)
    knn_feature_panel.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_knn_feature_panel.csv", index=False)
    knn_strategy_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_knn_strategy_returns.csv", index=False)
    knn_strategy_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_knn_strategy_summary.csv", index=False)
    logistic_feature_panel.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_logistic_feature_panel.csv", index=False)
    logistic_strategy_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_logistic_strategy_returns.csv", index=False)
    logistic_strategy_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_logistic_strategy_summary.csv", index=False)
    benchmark_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_benchmark_summary.csv", index=False)
    oos_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_oos_strategy_returns.csv", index=False)
    oos_choices.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_oos_model_selection.csv", index=False)
    oos_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_oos_strategy_summary.csv", index=False)
    weekly_strategy_panel.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_strategy_panel.csv", index=False)
    weekly_strategy_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_strategy_returns.csv", index=False)
    weekly_strategy_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_strategy_summary.csv", index=False)
    weekly_knn_feature_panel.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_knn_feature_panel.csv", index=False)
    weekly_knn_strategy_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_knn_strategy_returns.csv", index=False)
    weekly_knn_strategy_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_knn_strategy_summary.csv", index=False)
    weekly_logistic_feature_panel.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_logistic_feature_panel.csv", index=False)
    weekly_logistic_strategy_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_logistic_strategy_returns.csv", index=False)
    weekly_logistic_strategy_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_logistic_strategy_summary.csv", index=False)
    weekly_oos_returns.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_oos_strategy_returns.csv", index=False)
    weekly_oos_choices.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_oos_model_selection.csv", index=False)
    weekly_oos_summary.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_weekly_oos_strategy_summary.csv", index=False)
    write_method_note(validation_data)

    plot_proxy_validation(validation_summary)
    plot_rolling_r2(rolling_validation)
    plot_strategy_vs_spy(strategy_summary, strategy_returns, etf_returns)
    plot_k_sensitivity(strategy_summary)
    plot_knn_strategy_vs_spy(
        knn_strategy_summary,
        knn_strategy_returns,
        etf_returns,
        f"{OUTPUT_PREFIX}_knn_strategy_vs_spy.png",
        f"Daily KNN A/B Strategy Wealth vs SPY, Cost={PRIMARY_COST_BPS} bps",
    )
    plot_knn_k_sensitivity(
        knn_strategy_summary,
        f"{OUTPUT_PREFIX}_knn_k_sensitivity_heatmap.png",
        f"Daily KNN Sharpe by Factor and Neighbors, Cost={PRIMARY_COST_BPS} bps",
    )
    plot_knn_strategy_vs_spy(
        logistic_strategy_summary,
        logistic_strategy_returns,
        etf_returns,
        f"{OUTPUT_PREFIX}_logistic_strategy_vs_spy.png",
        f"Daily Logistic A/B Strategy Wealth vs SPY, Cost={PRIMARY_COST_BPS} bps",
    )
    plot_weekly_strategy_vs_spy(weekly_strategy_summary, weekly_strategy_returns, weekly_spy_returns)
    plot_weekly_k_sensitivity(weekly_strategy_summary)
    plot_knn_strategy_vs_spy(
        weekly_knn_strategy_summary,
        weekly_knn_strategy_returns,
        weekly_spy_returns,
        f"{OUTPUT_PREFIX}_weekly_knn_strategy_vs_spy.png",
        f"Weekly KNN A/B Strategy Wealth vs SPY, Cost={PRIMARY_COST_BPS} bps",
    )
    plot_knn_k_sensitivity(
        weekly_knn_strategy_summary,
        f"{OUTPUT_PREFIX}_weekly_knn_k_sensitivity_heatmap.png",
        f"Weekly KNN Sharpe by Factor and Neighbors, Cost={PRIMARY_COST_BPS} bps",
    )
    plot_knn_strategy_vs_spy(
        weekly_logistic_strategy_summary,
        weekly_logistic_strategy_returns,
        weekly_spy_returns,
        f"{OUTPUT_PREFIX}_weekly_logistic_strategy_vs_spy.png",
        f"Weekly Logistic A/B Strategy Wealth vs SPY, Cost={PRIMARY_COST_BPS} bps",
    )

    print("Saved publication analysis outputs to outputs/")
    print("\nProxy validation:")
    print(validation_summary[["factor", "proxy_definition", "correlation", "r_squared", "beta", "raw_tracking_error"]].to_string(index=False))
    print("\nPrimary strategy summary, k=13 Losers, 2 bps:")
    primary = strategy_summary[
        (strategy_summary["k"] == PRIMARY_K)
        & (strategy_summary["strategy_type"] == "Losers")
        & (strategy_summary["transaction_cost_bps"] == PRIMARY_COST_BPS)
    ]
    print(primary[["factor", "strategy_name", "final_wealth", "CAGR", "sharpe", "max_drawdown", "number_of_switches", "spy_alpha_annualized", "spy_beta"]].to_string(index=False))
    print("\nWeekly primary strategy summary, k=13 Losers, 2 bps:")
    weekly_primary = weekly_strategy_summary[
        (weekly_strategy_summary["k"] == PRIMARY_K)
        & (weekly_strategy_summary["strategy_type"] == "Losers")
        & (weekly_strategy_summary["transaction_cost_bps"] == PRIMARY_COST_BPS)
    ]
    print(weekly_primary[["factor", "strategy_name", "final_wealth", "CAGR", "sharpe", "max_drawdown", "number_of_switches", "spy_alpha_annualized", "spy_beta"]].to_string(index=False))
    print("\nDaily KNN best-by-factor summary, 2 bps:")
    daily_knn_best = (
        knn_strategy_summary.sort_values("sharpe", ascending=False)
        .groupby("factor")
        .head(1)
        .sort_values("factor")
    )
    print(daily_knn_best[["factor", "strategy_name", "final_wealth", "CAGR", "sharpe", "max_drawdown", "number_of_switches", "spy_alpha_annualized", "spy_beta"]].to_string(index=False))
    print("\nWeekly KNN best-by-factor summary, 2 bps:")
    weekly_knn_best = (
        weekly_knn_strategy_summary.sort_values("sharpe", ascending=False)
        .groupby("factor")
        .head(1)
        .sort_values("factor")
    )
    print(weekly_knn_best[["factor", "strategy_name", "final_wealth", "CAGR", "sharpe", "max_drawdown", "number_of_switches", "spy_alpha_annualized", "spy_beta"]].to_string(index=False))
    print("\nDaily logistic regression summary, 2 bps:")
    print(logistic_strategy_summary[["factor", "strategy_name", "final_wealth", "CAGR", "sharpe", "max_drawdown", "number_of_switches", "spy_alpha_annualized", "spy_beta"]].sort_values("factor").to_string(index=False))
    print("\nWeekly logistic regression summary, 2 bps:")
    print(weekly_logistic_strategy_summary[["factor", "strategy_name", "final_wealth", "CAGR", "sharpe", "max_drawdown", "number_of_switches", "spy_alpha_annualized", "spy_beta"]].sort_values("factor").to_string(index=False))


if __name__ == "__main__":
    run_publication_analysis()
