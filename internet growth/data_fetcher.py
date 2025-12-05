"""
Utilities for fetching or synthesizing internet usage data across regions.

The functions in this module avoid external network dependencies by allowing
callers to supply their own data provider. When no provider is supplied we
synthesize plausible historical penetration rates using a simple logistic
assumption so that downstream projections can be exercised in offline
environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence
import datetime as _dt
import random


@dataclass
class UsageRecord:
    """Represents the percentage of the population using the internet in a year.

    Attributes
    ----------
    region: str
        Region name (e.g., "North America", "Sub-Saharan Africa").
    year: int
        Calendar year of the observation.
    penetration: float
        Estimated share of population with internet access, between 0 and 1.
    """

    region: str
    year: int
    penetration: float


Provider = Callable[[Sequence[str], int, int], Iterable[UsageRecord]]


def _synthesize_penetration(year: int, start_year: int, baseline: float, ceiling: float) -> float:
    """Generate a deterministic but non-linear penetration rate.

    The function uses a logistic-like curve to keep values between 0 and a
    region-specific ceiling.
    """

    elapsed = max(year - start_year, 0)
    growth_rate = 0.18
    logistic = ceiling / (1 + (ceiling / baseline - 1) * pow(2.718, -growth_rate * elapsed))
    return min(max(logistic, 0.0), ceiling)


def synthesize_usage_data(
    regions: Sequence[str], start_year: int, end_year: int, seed: int | None = None
) -> List[UsageRecord]:
    """Create deterministic synthetic data for multiple regions.

    The function defaults to using a fixed seed so repeated invocations produce
    identical datasets. Penetration ceilings vary by region so that projections
    can be tested against heterogeneous adoption trajectories.
    """

    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")

    rng = random.Random(seed or 42)
    records: List[UsageRecord] = []
    now = _dt.datetime.utcnow().year

    for region in regions:
        baseline = rng.uniform(0.02, 0.20)
        ceiling = rng.uniform(0.75, 0.98)
        first_year = max(min(start_year, now), 1980)
        for year in range(first_year, end_year + 1):
            penetration = _synthesize_penetration(year, first_year, baseline, ceiling)
            records.append(UsageRecord(region=region, year=year, penetration=penetration))

    return records


def fetch_usage_data(
    regions: Sequence[str],
    start_year: int,
    end_year: int,
    provider: Provider | None = None,
    seed: int | None = None,
) -> List[UsageRecord]:
    """Fetch internet usage data for the requested regions.

    Parameters
    ----------
    regions:
        Collection of region names to include.
    start_year, end_year:
        Inclusive year bounds for the data series.
    provider:
        Optional callable for retrieving data from an external system. When not
        provided, synthetic data is generated for offline testing.
    seed:
        Optional seed for reproducible synthetic data when no provider is used.
    """

    if provider is None:
        return synthesize_usage_data(regions, start_year, end_year, seed=seed)

    return list(provider(regions, start_year, end_year))


def pivot_by_region(records: Iterable[UsageRecord]) -> Dict[str, List[UsageRecord]]:
    """Organize usage records into a lookup keyed by region."""

    pivot: Dict[str, List[UsageRecord]] = {}
    for record in records:
        pivot.setdefault(record.region, []).append(record)

    for region_records in pivot.values():
        region_records.sort(key=lambda r: r.year)

    return pivot
