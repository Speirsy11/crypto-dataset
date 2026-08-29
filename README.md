# Crypto Dataset

A public, research-friendly OHLCV crypto candle dataset built from Binance public 1-minute market data.

## Symbols

`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `TRXUSDT`, `DOGEUSDT`, `ZECUSDT`, `ADAUSDT`, `BCHUSDT`

## Intervals

`1m`, `5m`, `15m`, `1h`, `4h`, `1d`, `1w`, `1mo`

All candles use **UTC** boundaries. The weekly interval starts on Monday UTC. Monthly candles use calendar months UTC.

## Layout

```text
data/
  interval_id=1m/
    symbol_id=BTCUSDT/
      year=2026/
        month=05/
          BTCUSDT-1m-2026-05.parquet
metadata/
  manifest.json
```

Parquet files are partitioned by interval, symbol, year, and month. Aggregated candles include a `source_rows` column showing how many 1-minute rows were used.

## Columns

- `symbol`
- `interval`
- `timestamp` — candle open time in UTC
- `open`
- `high`
- `low`
- `close`
- `volume`
- `source_rows` — `1` for 1m rows; number of 1m candles rolled up for higher intervals

## Reading the data

Install dependencies:

```bash
python -m pip install pyarrow pandas
```

Read one Parquet file:

```python
import pandas as pd

path = "data/interval_id=1h/symbol_id=BTCUSDT/year=2026/month=05/BTCUSDT-1h-2026-05.parquet"
df = pd.read_parquet(path)
print(df.head())
```

Read a full symbol/interval directory:

```python
import pyarrow.dataset as ds

candles = ds.dataset("data/interval_id=1h/symbol_id=BTCUSDT", format="parquet", partitioning="hive")
df = candles.to_table().to_pandas().sort_values("timestamp")
print(df.tail())
```

Export back to CSV:

```python
df.to_csv("BTCUSDT-1h.csv", index=False)
```

## Updating

The dataset is generated, not hand-written. Use:

```bash
python scripts/build_dataset.py --publish-latest-day
```

That command detects the latest complete UTC day available in Signal Harvester, regenerates the affected partitions, updates `metadata/manifest.json`, commits, and pushes if there are changes.

For a full rebuild:

```bash
python scripts/build_dataset.py --full --cutoff-utc 2026-05-20T00:00:00Z
```

## Licensing

- Code/scripts: MIT (`LICENSE`)
- Data: CC0/public domain dedication (`DATA_LICENSE`)

Source data comes from Binance public market-data endpoints. This repository is not affiliated with Binance.

News and macro sentiment data now lives separately in
[crypto-news-dataset](https://github.com/Speirsy11/crypto-news-dataset).

<!-- AUTO-STATS START -->
## Dataset Stats

_Auto-generated on each publish — do not edit manually._

**Last generated:** 2026-08-29T23:26:09.297134Z
**Latest complete UTC day:** 2026-08-28
**Coverage:** 2017-08-17 → 2026-08-28

| Metric | Value |
|--------|-------|
| Symbols | 10 |
| Intervals | 8 |
| Parquet files | 7,630 |
| Total 1m candles | 41,492,433 |

**Per-symbol 1m candle counts:**

| Symbol | Candles | Earliest |
|--------|---------|----------|
| BTCUSDT | 4,732,812 | 2017-08-17 |
| ETHUSDT | 4,732,816 | 2017-08-17 |
| SOLUSDT | 3,170,287 | 2020-08-11 |
| BNBUSDT | 4,616,448 | 2017-11-06 |
| XRPUSDT | 4,360,733 | 2018-05-04 |
| TRXUSDT | 4,305,813 | 2018-06-11 |
| DOGEUSDT | 3,748,431 | 2019-07-05 |
| ZECUSDT | 3,900,657 | 2019-03-21 |
| ADAUSDT | 4,385,620 | 2018-04-17 |
| BCHUSDT | 3,538,816 | 2019-11-28 |

**Latest day (2026-08-28):**

| Metric | Value |
|--------|-------|
| 1m candles | 14,400 |
| Files updated | 10 |
<!-- AUTO-STATS END -->







































































































