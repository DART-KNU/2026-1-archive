from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline


DEFAULT_INPUT = Path("data/bigkinds_news_titles_20230510_20260510.csv")
DEFAULT_OUTPUT = Path("data/bigkinds_news_titles_sentiment.csv")
DEFAULT_DAILY_OUTPUT = Path("data/bigkinds_daily_sentiment_index.csv")
DEFAULT_MODEL = "snunlp/KR-FinBert-SC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis on BigKinds news titles."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Analyze only the first N rows. Use 0 for all rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the output CSV.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate titles. By default, duplicate titles are removed.",
    )
    return parser.parse_args()


def load_titles(path: Path, limit: int = 0, keep_duplicates: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date", "title"], dtype={"date": str, "title": str})
    df["date"] = df["date"].str.strip()
    df["title"] = df["title"].map(canonical_title)
    df = df[df["date"].notna() & df["title"].ne("")]
    df = df.sort_values(["date", "title"], kind="stable")
    if not keep_duplicates:
        df = df.drop_duplicates(subset=["title"], keep="first")
    if limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    done: set[tuple[str, str]] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row.get("date", "")
            title = row.get("title", "")
            if date and title:
                done.add((date, title))
    return done


def canonical_title(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def load_done_titles(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            title = row.get("title", "")
            if title:
                done.add(title)
    return done


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    if value in {"positive", "pos", "긍정"}:
        return "positive"
    if value in {"negative", "neg", "부정"}:
        return "negative"
    if value in {"neutral", "neu", "중립"}:
        return "neutral"

    # Fallback for models that expose only LABEL_0/LABEL_1/LABEL_2.
    label_map = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }
    return label_map.get(value, value)


def signed_score(sentiment: str, score: float) -> float:
    if sentiment == "positive":
        return score
    if sentiment == "negative":
        return -score
    return 0.0


def make_analyzer(model_name: str, max_length: int):
    device = 0 if torch.cuda.is_available() else -1
    print(f"loading model: {model_name}")
    print(f"device: {'cuda' if device == 0 else 'cpu'}")
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        max_length=max_length,
    )


def append_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "title", "sentiment", "score", "signed_score", "raw_label"]
    should_write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_daily_index(sentiment_output: Path, daily_output: Path) -> pd.DataFrame:
    df = pd.read_csv(sentiment_output, dtype={"date": str, "title": str, "sentiment": str})
    df["title"] = df["title"].map(canonical_title)
    df = df.drop_duplicates(subset=["title"], keep="first")
    grouped = defaultdict(Counter)

    for row in df.itertuples(index=False):
        grouped[row.date][row.sentiment] += 1

    rows: list[dict[str, int | float | str]] = []
    for date in sorted(grouped):
        counts = grouped[date]
        positive = counts["positive"]
        negative = counts["negative"]
        neutral = counts["neutral"]
        total = sum(counts.values())
        sentiment_index = (positive - negative) / total if total else 0.0
        rows.append(
            {
                "date": date,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "total": total,
                "sentiment_index": sentiment_index,
            }
        )

    daily_df = pd.DataFrame(rows)
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(daily_output, index=False, encoding="utf-8-sig")
    return daily_df


def analyze_titles(args: argparse.Namespace) -> None:
    df = load_titles(args.input, args.limit, args.keep_duplicates)
    if args.resume:
        done = load_done_titles(args.output)
        df = df[~df["title"].isin(done)].reset_index(drop=True)
        print(f"resume: skipped {len(done)} existing rows")

    print(f"rows to analyze: {len(df)}")
    if df.empty:
        write_daily_index(args.output, args.daily_output)
        return

    analyzer = make_analyzer(args.model, args.max_length)
    total = len(df)

    for start in range(0, total, args.batch_size):
        batch = df.iloc[start : start + args.batch_size]
        predictions = analyzer(batch["title"].tolist(), batch_size=args.batch_size)
        rows = []

        for source, prediction in zip(batch.itertuples(index=False), predictions):
            raw_label = str(prediction["label"])
            score = float(prediction["score"])
            sentiment = normalize_label(raw_label)
            rows.append(
                {
                    "date": source.date,
                    "title": source.title,
                    "sentiment": sentiment,
                    "score": score,
                    "signed_score": signed_score(sentiment, score),
                    "raw_label": raw_label,
                }
            )

        append_rows(args.output, rows)
        done_count = min(start + args.batch_size, total)
        print(f"processed {done_count}/{total}")

    daily_df = write_daily_index(args.output, args.daily_output)
    print(f"saved row-level sentiment: {args.output}")
    print(f"saved daily sentiment index: {args.daily_output}")
    print(f"daily rows: {len(daily_df)}")


def main() -> None:
    analyze_titles(parse_args())


if __name__ == "__main__":
    main()
