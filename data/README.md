# Data

## Included (real, committed to this repo)

- `btc_funding_binance.csv`, `eth_funding_binance.csv` -- real Binance
  perpetual-futures funding-rate settlement history, 2020-01-01 to
  2023-12-31, 8-hourly. Sourced from
  [supervik/historical-funding-rates-fetcher](https://github.com/supervik/historical-funding-rates-fetcher),
  itself a mirror of real exchange data.

## Not included (download these yourself, ~2 minutes)

Hourly OHLCV for BTCUSDT and ETHUSDT are ~5-10MB CSVs each and easy to
regenerate, so they're gitignored rather than committed. Get them free,
no signup, no API key:

```bash
curl -o data/btc_hourly.csv https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv
curl -o data/eth_hourly.csv https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_1h.csv
```

Then `python3 run_backtest.py` reproduces every number in the README.
