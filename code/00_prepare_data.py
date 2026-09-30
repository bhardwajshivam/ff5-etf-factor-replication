from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_FF5_PATH = DATA_DIR / "F-F_Research_Data_5_Factors_2x3_daily.csv"
FF5_OUTPUT_PATH = DATA_DIR / "ff5_cleaned_daily.csv"
BASE_ETF_OUTPUT_PATH = DATA_DIR / "etf_daily_returns.csv"
ALT_ETF_OUTPUT_PATH = DATA_DIR / "alt_etf_daily_returns.csv"

START_DATE = "2000-01-01"
REQUIRED_FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
BASE_ETF_TICKERS = ["IJR", "QQQ", "QUAL", "SPHB", "SPHQ", "SPY", "SPYG", "SPYV"]
ALT_ETF_TICKERS = ["ARKK", "IJR", "QUAL", "SPHB", "SPHQ", "SPLV", "SPY", "SPYG", "SPYV"]


def find_ff5_header_row(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for row_number, line in enumerate(file):
            normalized = line.replace(" ", "")
            if "Mkt-RF" in normalized and "SMB" in normalized and "HML" in normalized:
                return row_number
    raise ValueError(f"Could not find the Fama French header row in {path}")


def clean_ff5_daily_file(raw_path: Path = RAW_FF5_PATH, output_path: Path = FF5_OUTPUT_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw Fama French file: {raw_path}. Download the daily five factor CSV "
            "from the Kenneth R. French Data Library and place it at this path."
        )

    header_row = find_ff5_header_row(raw_path)
    data = pd.read_csv(raw_path, skiprows=header_row)
    data.columns = [str(column).strip() for column in data.columns]
    date_column = data.columns[0]
    data = data.rename(columns={date_column: "date"})

    missing = [column for column in REQUIRED_FF5_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required Fama French columns: {missing}")

    data["date"] = pd.to_datetime(data["date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    data = data.dropna(subset=["date"])
    data = data[["date", *REQUIRED_FF5_COLUMNS]].copy()
    for column in REQUIRED_FF5_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=REQUIRED_FF5_COLUMNS)
    data[REQUIRED_FF5_COLUMNS] = data[REQUIRED_FF5_COLUMNS] / 100.0
    data = data.rename(columns={"Mkt-RF": "MKT"}).sort_values("date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def download_adjusted_close_returns(tickers: list[str], output_path: Path) -> pd.DataFrame:
    downloaded = yf.download(
        tickers,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if downloaded.empty:
        raise ValueError("No ETF data downloaded from Yahoo Finance through yfinance.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        prices = downloaded["Close"].copy()
    else:
        prices = downloaded[["Close"]].copy()
        prices.columns = tickers

    missing = [ticker for ticker in tickers if ticker not in prices.columns or prices[ticker].dropna().empty]
    if missing:
        raise ValueError(f"Downloaded ETF data are missing expected tickers: {missing}")

    prices = prices.reindex(columns=tickers).sort_index().dropna(how="all")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns.index.name = "Date"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_path)
    return returns


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ff5 = clean_ff5_daily_file()
    base_returns = download_adjusted_close_returns(BASE_ETF_TICKERS, BASE_ETF_OUTPUT_PATH)
    alt_returns = download_adjusted_close_returns(ALT_ETF_TICKERS, ALT_ETF_OUTPUT_PATH)

    print(f"Saved {len(ff5):,} Fama French rows to {FF5_OUTPUT_PATH}")
    print(f"Saved {len(base_returns):,} base ETF return rows to {BASE_ETF_OUTPUT_PATH}")
    print(f"Saved {len(alt_returns):,} alternative ETF return rows to {ALT_ETF_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
