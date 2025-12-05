"""
Model utilities for projecting internet usage growth across regions.

The functions operate on the UsageRecord objects defined in ``data_fetcher`` and
implement a lightweight compounded growth projection that tapers as adoption
approaches a configurable saturation level.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.append(str(MODULE_ROOT))

from data_fetcher import UsageRecord, pivot_by_region


@dataclass
class GrowthModelResult:
    """Projected adoption for a single region."""

    region: str
    baseline_year: int
    horizon_year: int
    projections: List[UsageRecord]


@dataclass
class ModelSummary:
    """Aggregate output across all modelled regions."""

    method: str
    results: List[GrowthModelResult]


def _calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """Compute the compound annual growth rate for penetration values."""

    if start_value <= 0 or periods <= 0:
        return 0.0
    return pow(end_value / start_value, 1 / periods) - 1


def _project_region(
    records: Sequence[UsageRecord], years_ahead: int, saturation: float
) -> GrowthModelResult:
    ordered = sorted(records, key=lambda r: r.year)
    first, last = ordered[0], ordered[-1]
    cagr = _calculate_cagr(first.penetration, last.penetration, last.year - first.year or 1)

    projections: List[UsageRecord] = []
    prior_value = last.penetration
    for offset in range(1, years_ahead + 1):
        projected = min(prior_value * (1 + cagr), saturation)
        year = last.year + offset
        projections.append(UsageRecord(region=last.region, year=year, penetration=projected))
        prior_value = projected

    return GrowthModelResult(
        region=last.region,
        baseline_year=last.year,
        horizon_year=last.year + years_ahead,
        projections=projections,
    )


def build_growth_model(
    records: Iterable[UsageRecord], years_ahead: int = 5, saturation: float = 0.97
) -> ModelSummary:
    """Project internet usage for multiple regions.

    Parameters
    ----------
    records:
        Historical UsageRecord entries for one or more regions.
    years_ahead:
        Number of years to project beyond the latest observation.
    saturation:
        Upper bound for adoption percentage to avoid unrealistic growth.
    """

    if not 0 < saturation <= 1:
        raise ValueError("saturation must be between 0 and 1")

    pivot = pivot_by_region(records)
    results = [_project_region(region_records, years_ahead=years_ahead, saturation=saturation) for region_records in pivot.values()]

    return ModelSummary(method="compound_growth_projection", results=results)
