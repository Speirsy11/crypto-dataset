#!/usr/bin/env python3
"""Score private feed text, delete it, and publish metadata-only sentiment Parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sentiment"
DAILY_DIR = ROOT / "sentiment_daily"
MANIFEST_PATH = ROOT / "metadata" / "sentiment-manifest.json"
DEFAULT_DATABASE_URL = "postgresql://signal:signal@localhost:5544/signal_harvester"
MODEL_NAME = os.environ.get("SENTIMENT_MODEL", "ProsusAI/finbert")
MODEL_REVISION = os.environ.get("SENTIMENT_MODEL_REVISION", "main")
HEADLINE_WEIGHT = 0.35
BODY_WEIGHT = 0.65

ARTICLE_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("source_type", pa.string()),
    ("source_name", pa.string()),
    ("asset", pa.string()),
    ("relevance_score", pa.float32()),
    ("headline", pa.string()),
    ("url", pa.string()),
    ("author", pa.string()),
    ("published_at", pa.timestamp("us", tz="UTC")),
    ("collected_at", pa.timestamp("us", tz="UTC")),
    ("headline_positive", pa.float32()),
    ("headline_neutral", pa.float32()),
    ("headline_negative", pa.float32()),
    ("body_positive", pa.float32()),
    ("body_neutral", pa.float32()),
    ("body_negative", pa.float32()),
    ("combined_positive", pa.float32()),
    ("combined_neutral", pa.float32()),
    ("combined_negative", pa.float32()),
    ("combined_score", pa.float32()),
    ("combined_label", pa.string()),
    ("lexicon_score", pa.float32()),
    ("lexicon_label", pa.string()),
    ("model_name", pa.string()),
    ("model_version", pa.string()),
    ("processed_at", pa.timestamp("us", tz="UTC")),
    ("input_sha256", pa.string()),
])

DAILY_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("source_type", pa.string()),
    ("asset", pa.string()),
    ("document_count", pa.int32()),
    ("source_count", pa.int32()),
    ("relevance_weight", pa.float64()),
    ("weighted_sentiment", pa.float64()),
    ("positive_share", pa.float64()),
    ("neutral_share", pa.float64()),
    ("negative_share", pa.float64()),
    ("model_name", pa.string()),
    ("model_version", pa.string()),
])


def database_url() -> str:
    return os.environ.get("SIGNAL_HARVESTER_DATABASE_URL", DEFAULT_DATABASE_URL)


def source_type(topic: str, source_name: str) -> str:
    if topic == "MACRO":
        return "macro"
    if source_name == "Reddit CryptoCurrency":
        return "public_discussion"
    return "news"


def load_model():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "FinBERT dependencies missing; install requirements.txt in the dataset virtualenv"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.eval()
    return torch, tokenizer, model


def score_text(text: str, runtime) -> dict[str, float]:
    torch, tokenizer, model = runtime
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if not encoded:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    chunks = [encoded[start:start + 510] for start in range(0, len(encoded), 480)]
    totals = defaultdict(float)
    total_weight = 0
    with torch.inference_mode():
        for chunk in chunks:
            token_ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
            inputs = {
                "input_ids": torch.tensor([token_ids]),
                "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
            }
            probabilities = torch.softmax(model(**inputs).logits[0], dim=-1).tolist()
            weight = len(chunk)
            for index, probability in enumerate(probabilities):
                label = str(model.config.id2label[index]).lower()
                totals[label] += probability * weight
            total_weight += weight
    return {
        label: totals[label] / total_weight
        for label in ("positive", "neutral", "negative")
    }


def combined_result(headline: dict[str, float], body: dict[str, float] | None):
    if body is None:
        probabilities = headline
    else:
        probabilities = {
            label: headline[label] * HEADLINE_WEIGHT + body[label] * BODY_WEIGHT
            for label in ("positive", "neutral", "negative")
        }
    score = probabilities["positive"] - probabilities["negative"]
    label = max(probabilities, key=probabilities.get)
    return probabilities, score, label


def process_pending(limit: int | None = None) -> int:
    runtime = load_model()
    model_version = getattr(runtime[2].config, "_commit_hash", None) or MODEL_REVISION
    processed = 0
    with psycopg.connect(database_url(), row_factory=dict_row) as conn:
        params: list[object] = [MODEL_NAME, model_version]
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT %s"
            params.append(limit)
        rows = conn.execute(
            f"""
            SELECT d.*
            FROM documents d
            WHERE NOT EXISTS (
              SELECT 1 FROM document_sentiments s
              WHERE s.document_id = d.id
                AND s.model_name = %s
                AND s.model_version = %s
            )
            ORDER BY (d.private_content IS NOT NULL) DESC,
                     COALESCE(d.published_at, d.collected_at) DESC,
                     d.id
            {limit_sql}
            """,
            params,
        ).fetchall()

        for row in rows:
            title = row["title"]
            body = row["private_content"] or row["summary"]
            headline_probs = score_text(title, runtime)
            body_probs = score_text(body, runtime) if body else None
            combined_probs, score, label = combined_result(headline_probs, body_probs)
            digest = hashlib.sha256(f"{title}\n{body or ''}".encode()).hexdigest()
            conn.execute(
                """
                INSERT INTO document_sentiments (
                  document_id, model_name, model_version,
                  headline_positive, headline_neutral, headline_negative,
                  body_positive, body_neutral, body_negative,
                  combined_positive, combined_neutral, combined_negative,
                  combined_score, combined_label, input_sha256
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (document_id, model_name, model_version) DO NOTHING
                """,
                (
                    row["id"], MODEL_NAME, model_version,
                    headline_probs["positive"], headline_probs["neutral"], headline_probs["negative"],
                    body_probs["positive"] if body_probs else None,
                    body_probs["neutral"] if body_probs else None,
                    body_probs["negative"] if body_probs else None,
                    combined_probs["positive"], combined_probs["neutral"], combined_probs["negative"],
                    score, label, digest,
                ),
            )
            conn.execute(
                "UPDATE documents SET private_content = NULL, content_delete_after = NULL WHERE id = %s",
                (row["id"],),
            )
            conn.commit()
            processed += 1

        purged = conn.execute(
            """
            UPDATE documents
            SET private_content = NULL, content_delete_after = NULL
            WHERE private_content IS NOT NULL AND content_delete_after <= NOW()
            """,
        ).rowcount
        conn.commit()
        if purged:
            print(f"Purged {purged} expired private content rows")
    print(f"Processed {processed} documents with {MODEL_NAME}")
    return processed


def article_path(row) -> Path:
    timestamp = row["published_at"] or row["collected_at"]
    kind = source_type(row["topic"], row["source_name"])
    return (
        DATA_DIR / f"source_type={kind}" / f"asset={row['asset']}"
        / f"year={timestamp.year:04d}" / f"month={timestamp.month:02d}"
        / f"day={timestamp.day:02d}" / "sentiment.parquet"
    )


def rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def export_dataset() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(database_url(), row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT
              d.id::text AS document_id, d.topic, d.source_name, r.asset,
              r.relevance_score, d.title AS headline, d.url, d.author,
              COALESCE(d.published_at, d.collected_at) AS published_at,
              d.collected_at,
              s.headline_positive, s.headline_neutral, s.headline_negative,
              s.body_positive, s.body_neutral, s.body_negative,
              s.combined_positive, s.combined_neutral, s.combined_negative,
              s.combined_score, s.combined_label,
              d.sentiment_score AS lexicon_score, d.sentiment_label AS lexicon_label,
              s.model_name, s.model_version, s.processed_at, s.input_sha256
            FROM documents d
            JOIN document_asset_relevance r ON r.document_id = d.id
            JOIN LATERAL (
              SELECT *
              FROM document_sentiments candidate
              WHERE candidate.document_id = d.id
              ORDER BY candidate.processed_at DESC
              LIMIT 1
            ) s ON true
            ORDER BY COALESCE(d.published_at, d.collected_at), d.id, r.asset
            """
        ).fetchall()

    partitions: dict[Path, list[dict]] = defaultdict(list)
    for row in rows:
        output = dict(row)
        output["source_type"] = source_type(row["topic"], row["source_name"])
        output.pop("topic")
        partitions[article_path(row)].append(output)

    for path, partition_rows in partitions.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows_to_table(partition_rows, ARTICLE_SCHEMA), path, compression="zstd")

    daily_groups: dict[tuple[date, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        timestamp = row["published_at"] or row["collected_at"]
        kind = source_type(row["topic"], row["source_name"])
        key = (timestamp.date(), kind, row["asset"], row["model_name"], row["model_version"])
        daily_groups[key].append(row)

    daily_by_path: dict[Path, list[dict]] = defaultdict(list)
    for (day, kind, asset, model_name, model_version), group in daily_groups.items():
        weight = sum(float(row["relevance_score"]) for row in group)
        weighted = lambda field: sum(float(row[field]) * float(row["relevance_score"]) for row in group) / weight
        daily = {
            "date": day,
            "source_type": kind,
            "asset": asset,
            "document_count": len(group),
            "source_count": len({row["source_name"] for row in group}),
            "relevance_weight": weight,
            "weighted_sentiment": weighted("combined_score"),
            "positive_share": weighted("combined_positive"),
            "neutral_share": weighted("combined_neutral"),
            "negative_share": weighted("combined_negative"),
            "model_name": model_name,
            "model_version": model_version,
        }
        path = (
            DAILY_DIR / f"source_type={kind}" / f"asset={asset}"
            / f"year={day.year:04d}" / f"month={day.month:02d}" / "daily.parquet"
        )
        daily_by_path[path].append(daily)

    for path, partition_rows in daily_by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        partition_rows.sort(key=lambda row: row["date"])
        pq.write_table(rows_to_table(partition_rows, DAILY_SCHEMA), path, compression="zstd")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({
        "name": "crypto-and-macro-sentiment",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": MODEL_NAME,
        "requested_model_revision": MODEL_REVISION,
        "headline_weight": HEADLINE_WEIGHT,
        "body_weight": BODY_WEIGHT,
        "documents": len(rows),
        "privacy": "Feed content is processed privately and is never exported.",
        "article_layout": "sentiment/source_type=<type>/asset=<asset>/year=<YYYY>/month=<MM>/day=<DD>/sentiment.parquet",
        "daily_layout": "sentiment_daily/source_type=<type>/asset=<asset>/year=<YYYY>/month=<MM>/daily.parquet",
    }, indent=2) + "\n")
    print(f"Exported {len(rows)} document-asset sentiment rows")
    return len(rows)


def commit_and_push() -> None:
    paths = [
        path for path in
        ["README.md", "requirements.txt", "scripts", "metadata", "sentiment", "sentiment_daily"]
        if (ROOT / path).exists()
    ]
    subprocess.run(
        ["git", "add", *paths],
        cwd=ROOT, check=True,
    )
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if not status.strip():
        print("No sentiment dataset changes to commit")
        return
    day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    subprocess.run(["git", "commit", "-m", f"Update sentiment dataset through {day}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push"], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not (args.process or args.export or args.publish):
        parser.error("choose --process, --export, or --publish")
    if args.process or args.publish:
        process_pending(args.limit)
    if args.export or args.publish:
        export_dataset()
    if args.publish:
        commit_and_push()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
