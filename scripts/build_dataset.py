#!/usr/bin/env python3
"""Build and publish the Speirsy11 crypto Parquet dataset from Signal Harvester Postgres."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = ROOT / "metadata" / "manifest.json"
README_PATH = ROOT / "README.md"
README_STATS_START = "<!-- AUTO-STATS START -->"
README_STATS_END = "<!-- AUTO-STATS END -->"
DEFAULT_DATABASE_URL = "postgresql://signal:signal@localhost:5544/signal_harvester"
PROVIDER = "binance"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "TRXUSDT", "DOGEUSDT", "ZECUSDT", "ADAUSDT", "BCHUSDT"]
INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"]
INTERVAL_SQL = {
    "5m": "5 minutes",
    "15m": "15 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week",
    "1mo": "1 month",
}
DATE_BIN_INTERVALS = {"5m", "15m", "1h", "4h"}
SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("interval", pa.string()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("source_rows", pa.int32()),
])


@dataclass(frozen=True)
class Month:
    start: datetime

    @property
    def end(self) -> datetime:
        year = self.start.year + (1 if self.start.month == 12 else 0)
        month = 1 if self.start.month == 12 else self.start.month + 1
        return datetime(year, month, 1, tzinfo=timezone.utc)

    @property
    def year(self) -> int:
        return self.start.year

    @property
    def month(self) -> int:
        return self.start.month


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def month_floor(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    return datetime(value.year, value.month, 1, tzinfo=timezone.utc)


def iter_months(start: datetime, end_exclusive: datetime) -> Iterable[Month]:
    current = month_floor(start)
    stop = month_floor(end_exclusive - timedelta(microseconds=1))
    while current <= stop:
        yield Month(current)
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = datetime(year, month, 1, tzinfo=timezone.utc)


def parquet_path(symbol: str, interval: str, month: Month) -> Path:
    return DATA_DIR / f"interval_id={interval}" / f"symbol_id={symbol}" / f"year={month.year:04d}" / f"month={month.month:02d}" / f"{symbol}-{interval}-{month.year:04d}-{month.month:02d}.parquet"


def table_from_rows(rows) -> pa.Table:
    columns = {name: [] for name in SCHEMA.names}
    for row in rows:
        for name in SCHEMA.names:
            columns[name].append(row[name])
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def write_table(path: Path, table: pa.Table) -> bool:
    if table.num_rows == 0:
        if path.exists():
            path.unlink()
            return True
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=9)
    return True


def fetch_symbol_bounds(conn) -> dict[str, datetime]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT symbol, min(timestamp) AS start_time
            FROM market_data_points
            WHERE provider = %s AND interval = '1m' AND symbol = ANY(%s)
            GROUP BY symbol
            """,
            (PROVIDER, SYMBOLS),
        )
        return {row["symbol"]: row["start_time"].astimezone(timezone.utc) for row in cur.fetchall()}


def latest_complete_day(conn) -> datetime:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date_trunc('day', min(latest_ts) + interval '1 minute')
            FROM (
              SELECT symbol, max(timestamp) AS latest_ts
              FROM market_data_points
              WHERE provider = %s AND interval = '1m' AND symbol = ANY(%s)
              GROUP BY symbol
            ) latest
            """,
            (PROVIDER, SYMBOLS),
        )
        value = cur.fetchone()[0]
    if value is None:
        raise RuntimeError("No source candles found")
    # max(timestamp) is a 1m candle open. Adding one minute gives the newest covered instant;
    # flooring that to a UTC day yields the exclusive cutoff for fully completed UTC days.
    return value.astimezone(timezone.utc)


def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def write_manifest(cutoff: datetime, bounds: dict[str, datetime]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "crypto-dataset",
        "source": "Binance public klines via Signal Harvester",
        "provider": PROVIDER,
        "timezone": "UTC",
        "symbols": SYMBOLS,
        "intervals": INTERVALS,
        "cutoff_utc_exclusive": cutoff.isoformat().replace("+00:00", "Z"),
        "last_complete_day_utc": (cutoff - timedelta(days=1)).date().isoformat(),
        "symbol_start_times": {k: v.isoformat().replace("+00:00", "Z") for k, v in sorted(bounds.items())},
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "layout": "data/interval_id=<interval>/symbol_id=<symbol>/year=<YYYY>/month=<MM>/<symbol>-<interval>-<YYYY>-<MM>.parquet",
        "notes": [
            "All timestamps are candle open times in UTC.",
            "Higher intervals are rolled up from 1m candles using UTC boundaries.",
            "Weekly candles start on Monday UTC.",
            "source_rows records the number of 1m rows used in each candle.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")




def _count_parquet_file_stats() -> dict:
    """Scan Parquet file footers for dataset statistics (footers only — no data loaded)."""
    if not DATA_DIR.exists():
        return {"total_files": 0, "total_1m_rows": 0, "per_symbol_1m": {}}
    total_files = 0
    total_1m_rows = 0
    per_symbol_1m: dict[str, int] = {}
    for interval_dir in sorted(DATA_DIR.iterdir()):
        if not interval_dir.is_dir():
            continue
        interval = interval_dir.name.removeprefix("interval_id=")
        for symbol_dir in sorted(interval_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name.removeprefix("symbol_id=")
            sym_1m = 0
            for pq_file in symbol_dir.rglob("*.parquet"):
                total_files += 1
                num_rows = pq.read_metadata(pq_file).num_rows
                if interval == "1m":
                    sym_1m += num_rows
            if interval == "1m":
                total_1m_rows += sym_1m
                per_symbol_1m[symbol] = sym_1m
    return {"total_files": total_files, "total_1m_rows": total_1m_rows, "per_symbol_1m": per_symbol_1m}


def _latest_day_1m_stats(day: date) -> dict:
    """Count 1m rows for a specific UTC day across all symbols."""
    month_start = datetime(day.year, day.month, 1, tzinfo=timezone.utc)
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    rows = 0
    files_touched = 0
    for symbol in SYMBOLS:
        path = parquet_path(symbol, "1m", Month(month_start))
        if not path.exists():
            continue
        table = pq.read_table(path, filters=[
            ("timestamp", ">=", day_start),
            ("timestamp", "<", day_end),
        ])
        rows += table.num_rows
        files_touched += 1
    return {"1m_rows": rows, "files_touched": files_touched}


def update_readme_stats() -> None:
    """Regenerate the auto-stats section of README.md from Parquet file footers and the manifest."""
    manifest = read_manifest()
    file_stats = _count_parquet_file_stats()
    last_day_str = manifest.get("last_complete_day_utc", "")
    latest_day_counts = {"1m_rows": 0, "files_touched": 0}
    if last_day_str:
        latest_day_counts = _latest_day_1m_stats(date.fromisoformat(last_day_str))
    start_times = manifest.get("symbol_start_times", {})
    earliest = min(start_times.values())[:10] if start_times else "N/A"
    per_sym = file_stats["per_symbol_1m"]
    symbol_rows = "\n".join(
        f"| {symbol} | {per_sym.get(symbol, 0):,} | {start_times.get(symbol, '')[:10] or 'N/A'} |"
        for symbol in SYMBOLS
    )
    stats_block = f"""{README_STATS_START}
## Dataset Stats

_Auto-generated on each publish — do not edit manually._

**Last generated:** {manifest.get("generated_at", "N/A")}
**Latest complete UTC day:** {last_day_str or "N/A"}
**Coverage:** {earliest} → {last_day_str or "N/A"}

| Metric | Value |
|--------|-------|
| Symbols | {len(SYMBOLS)} |
| Intervals | {len(INTERVALS)} |
| Parquet files | {file_stats["total_files"]:,} |
| Total 1m candles | {file_stats["total_1m_rows"]:,} |

**Per-symbol 1m candle counts:**

| Symbol | Candles | Earliest |
|--------|---------|----------|
{symbol_rows}

**Latest day ({last_day_str or "N/A"}):**

| Metric | Value |
|--------|-------|
| 1m candles | {latest_day_counts["1m_rows"]:,} |
| Files updated | {latest_day_counts["files_touched"]} |
{README_STATS_END}
"""
    text = README_PATH.read_text() if README_PATH.exists() else ""
    if README_STATS_START in text and README_STATS_END in text:
        before = text[:text.index(README_STATS_START)]
        after = text[text.index(README_STATS_END) + len(README_STATS_END):]
        README_PATH.write_text(before + stats_block + after)
    else:
        README_PATH.write_text(text.rstrip() + "\n\n" + stats_block)

def export_1m_partition(conn, symbol: str, month: Month, cutoff: datetime) -> bool:
    start = month.start
    end = min(month.end, cutoff)
    if end <= start:
        return False
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT symbol, '1m' AS interval, timestamp, open, high, low, close, volume, 1::int AS source_rows
            FROM market_data_points
            WHERE provider = %s AND symbol = %s AND interval = '1m'
              AND timestamp >= %s AND timestamp < %s
            ORDER BY timestamp
            """,
            (PROVIDER, symbol, start, end),
        )
        table = table_from_rows(cur.fetchall())
    return write_table(parquet_path(symbol, "1m", month), table)


def bucket_expr(interval: str) -> str:
    if interval in DATE_BIN_INTERVALS:
        return f"date_bin(interval '{INTERVAL_SQL[interval]}', timestamp, timestamptz '1970-01-01 00:00:00+00')"
    if interval == "1d":
        return "date_trunc('day', timestamp)"
    if interval == "1w":
        return "date_trunc('week', timestamp)"
    if interval == "1mo":
        return "date_trunc('month', timestamp)"
    raise ValueError(f"Unsupported interval: {interval}")


def export_aggregate_partition(conn, symbol: str, interval: str, month: Month, cutoff: datetime) -> bool:
    sql_interval = INTERVAL_SQL[interval]
    expr = bucket_expr(interval)
    input_end = min(month.end + timedelta(days=40), cutoff)
    if input_end <= month.start:
        return False
    query = f"""
        WITH binned AS (
          SELECT {expr} AS bucket, timestamp AS source_ts, open, high, low, close, volume
          FROM market_data_points
          WHERE provider = %s AND symbol = %s AND interval = '1m'
            AND timestamp >= %s AND timestamp < %s
        )
        SELECT
          %s AS symbol,
          %s AS interval,
          bucket AS timestamp,
          (array_agg(open ORDER BY source_ts ASC))[1]::double precision AS open,
          max(high)::double precision AS high,
          min(low)::double precision AS low,
          (array_agg(close ORDER BY source_ts DESC))[1]::double precision AS close,
          sum(volume)::double precision AS volume,
          count(*)::int AS source_rows
        FROM binned
        WHERE bucket >= %s AND bucket < %s AND bucket + interval '{sql_interval}' <= %s
        GROUP BY bucket
        ORDER BY bucket
    """
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query, (PROVIDER, symbol, month.start, input_end, symbol, interval, month.start, month.end, cutoff))
        table = table_from_rows(cur.fetchall())
    return write_table(parquet_path(symbol, interval, month), table)


def build_dataset(cutoff: datetime, changed_since: datetime | None = None) -> None:
    database_url = os.environ.get("SIGNAL_HARVESTER_DATABASE_URL", DEFAULT_DATABASE_URL)
    cutoff = cutoff.astimezone(timezone.utc)
    with psycopg.connect(database_url) as conn:
        bounds = fetch_symbol_bounds(conn)
        if not bounds:
            raise RuntimeError("No symbol bounds found")
        for symbol in SYMBOLS:
            start = bounds.get(symbol)
            if not start:
                print(f"WARN: no source candles for {symbol}", file=sys.stderr)
                continue
            export_start = max(start, changed_since) if changed_since else start
            print(f"{symbol}: exporting from {export_start.isoformat()} to {cutoff.isoformat()}")
            for month in iter_months(export_start, cutoff):
                for interval in INTERVALS:
                    if interval == "1m":
                        changed = export_1m_partition(conn, symbol, month, cutoff)
                    else:
                        changed = export_aggregate_partition(conn, symbol, interval, month, cutoff)
                    if changed:
                        print(f"  wrote {parquet_path(symbol, interval, month).relative_to(ROOT)}")
        write_manifest(cutoff, bounds)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check)


def commit_and_push(message: str) -> None:
    run(["git", "add", "README.md", "LICENSE", "DATA_LICENSE", "requirements.txt", ".gitignore", ".gitattributes", "scripts", "metadata", "data"])
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if not status.strip():
        print("No dataset changes to commit")
        return
    run(["git", "commit", "-m", message])
    run(["git", "push"])


def publish_latest_day() -> None:
    database_url = os.environ.get("SIGNAL_HARVESTER_DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(database_url) as conn:
        cutoff = latest_complete_day(conn)
    manifest = read_manifest()
    previous_cutoff = parse_utc(manifest["cutoff_utc_exclusive"]) if manifest.get("cutoff_utc_exclusive") else None
    if previous_cutoff and cutoff <= previous_cutoff:
        print(f"Dataset already current through {previous_cutoff.isoformat()}")
        return
    # Rebuild a conservative window so 1w/1mo aggregates closing after the new day are refreshed too.
    changed_since = (previous_cutoff - timedelta(days=40)) if previous_cutoff else None
    build_dataset(cutoff, changed_since=changed_since)
    update_readme_stats()
    commit_and_push(f"Update crypto dataset through {(cutoff - timedelta(days=1)).date().isoformat()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true", help="Build all partitions through --cutoff-utc")
    mode.add_argument("--publish-latest-day", action="store_true", help="Build and push the latest complete UTC day")
    parser.add_argument("--cutoff-utc", help="Exclusive UTC cutoff, e.g. 2026-05-19T23:00:00Z")
    parser.add_argument("--commit", action="store_true", help="Commit and push after --full")
    args = parser.parse_args()

    if args.full:
        if not args.cutoff_utc:
            raise SystemExit("--full requires --cutoff-utc")
        cutoff = parse_utc(args.cutoff_utc)
        build_dataset(cutoff)
        if args.commit:
            update_readme_stats()
            commit_and_push(f"Build crypto dataset through {(cutoff - timedelta(days=1)).date().isoformat()}")
    else:
        publish_latest_day()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
