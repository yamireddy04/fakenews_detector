"""
Dataset loaders for common fake news benchmarks.
All loaders normalize to: {"text": str, "body": str, "label": int, "source": str}
"""

import csv
import json
from pathlib import Path
from typing import Iterator
import logging

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "REAL", 1: "FAKE", 2: "UNVERIFIED"}

LIAR_LABEL_MAP = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 2,
    "barely-true": 1,
    "false": 1,
    "pants-fire": 1,
}


def load_liar(tsv_path: str | Path) -> list[dict]:
    """
    Columns: id, label, statement, subject, speaker, job, state, party,
             barely_true_count, false_count, half_true_count, mostly_true_count,
             pants_count, context
    """
    records = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            raw_label = row[1].strip()
            records.append(
                {
                    "text": row[2].strip(),
                    "body": "",
                    "label": LIAR_LABEL_MAP.get(raw_label, 2),
                    "source": "liar",
                    "speaker": row[4].strip() if len(row) > 4 else "",
                    "raw_label": raw_label,
                }
            )
    logger.info(f"LIAR: loaded {len(records)} records from {tsv_path}")
    return records

def load_fakenewsnet(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    records = []
    for domain in ["politifact", "gossipcop"]:
        for split, label in [("fake", 1), ("real", 0)]:
            split_dir = data_dir / domain / split
            if not split_dir.exists():
                continue
            for article_dir in split_dir.iterdir():
                content_file = article_dir / "news content.json"
                if not content_file.exists():
                    continue
                try:
                    with open(content_file, encoding="utf-8") as f:
                        data = json.load(f)
                    records.append(
                        {
                            "text": data.get("title", ""),
                            "body": data.get("text", "")[:2000],
                            "label": label,
                            "source": f"fakenewsnet-{domain}",
                            "url": data.get("url", ""),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Skipping {content_file}: {e}")
    logger.info(f"FakeNewsNet: loaded {len(records)} records from {data_dir}")
    return records

def load_clef(jsonl_path: str | Path) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = 1 if obj.get("class_label", "No") == "Yes" else 0
            records.append(
                {
                    "text": obj.get("tweet_text", obj.get("text", "")),
                    "body": "",
                    "label": label,
                    "source": "clef",
                    "lang": obj.get("lang", "en"),
                }
            )
    logger.info(f"CLEF: loaded {len(records)} records from {jsonl_path}")
    return records

def load_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    body_col: str = "",
    delimiter: str = ",",
) -> list[dict]:
    """
    Flexible loader for any CSV with text + label columns.
    Labels are auto-mapped: 0/false/real → 0=REAL, 1/true/fake → 1=FAKE
    """
    BOOL_MAP = {
        "0": 0, "false": 0, "real": 0, "legit": 0,
        "1": 1, "true": 1, "fake": 1, "FALSE": 1,
    }
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            raw = str(row[label_col]).strip().lower()
            label = BOOL_MAP.get(raw, 2)
            records.append(
                {
                    "text": row[text_col].strip(),
                    "body": row[body_col].strip() if body_col and body_col in row else "",
                    "label": label,
                    "source": "csv",
                }
            )
    return records

def split_records(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    import random
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return shuffled[:t], shuffled[t : t + v], shuffled[t + v :]