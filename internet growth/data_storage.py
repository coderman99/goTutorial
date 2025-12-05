"""
Persistence helpers for internet usage datasets and projections.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.append(str(MODULE_ROOT))

from data_fetcher import UsageRecord
from model_builder import ModelSummary


def ensure_directory(path: str | Path) -> Path:
    """Create parent directories for the provided file path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_usage_records_csv(records: Iterable[UsageRecord], path: str | Path) -> Path:
    """Persist usage records to a CSV file."""

    target = ensure_directory(path)
    fieldnames = ["region", "year", "penetration"]
    with target.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return target


def save_model_summary(summary: ModelSummary, path: str | Path) -> Path:
    """Persist model projections as JSON for downstream consumers."""

    target = ensure_directory(path)
    payload = {
        "method": summary.method,
        "results": [
            {
                "region": result.region,
                "baseline_year": result.baseline_year,
                "horizon_year": result.horizon_year,
                "projections": [asdict(projection) for projection in result.projections],
            }
            for result in summary.results
        ],
    }

    with target.open("w") as handle:
        json.dump(payload, handle, indent=2)

    return target
