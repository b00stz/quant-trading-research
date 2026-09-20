"""
Data loaders.

Funding rate CSVs: real data from
  github.com/supervik/historical-funding-rates-fetcher
  columns: Symbol, Date, Funding Rate   (8h settlement, Binance)

OHLCV CSVs: expected in CryptoDataDownload's Binance hourly-kline format
  columns: Unix, Date, Symbol, Open, High, Low, Close, Volume <BASE>,
           Volume USDT, tradecount
  (this is the format at
   https://www.cryptodatadownload.com/cdd/Binance_<PAIR>_1h.csv )
"""
import pandas as pd


def load_funding(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["Funding Rate"].rename("funding_rate")


def load_ohlcv_cdd(path: str) -> pd.DataFrame:
    with open(path) as f:
        first_line = f.readline()
    skiprows = 1 if first_line.strip().lower().startswith("http") else 0
    df = pd.read_csv(path, skiprows=skiprows)
    # Note: the Unix column has a x1000 units bug on a subset of rows in this
    # source file (extra trailing zeros push some timestamps to the year
    # 57163). The Date string column doesn't have that bug, so parse that
    # instead, with format='mixed' since a minority of rows carry a
    # ".000" fractional-second suffix that others don't.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date").set_index("Date")
    vol_col = [c for c in df.columns if c.startswith("Volume") and "USDT" not in c][0]
    out = pd.DataFrame({
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
        "volume": df[vol_col].astype(float),
        "quote_volume": df["Volume USDT"].astype(float),
        "trades": df["tradecount"].astype(float) if "tradecount" in df.columns else float("nan"),
    })
    # de-duplicate any repeated timestamps, keep last
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
