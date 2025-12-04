import datetime as dt
from typing import Tuple

import pandas as pd
import requests

COINGECKO_API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
FEAR_GREED_API = "https://api.alternative.me/fng/"


def _coingecko_params(days: int) -> dict:
    return {"vs_currency": "usd", "days": days, "interval": "hourly"}


def fetch_market_data(days: int = 3000) -> pd.DataFrame:
    """Fetch market data from CoinGecko and return daily closes.

    Args:
        days: Number of trailing days to request from CoinGecko.

    Returns:
        DataFrame with columns [date, close].
    """

    response = requests.get(COINGECKO_API, params=_coingecko_params(days), timeout=30)
    response.raise_for_status()
    prices = response.json().get("prices", [])

    if not prices:
        raise ValueError("No price data returned from CoinGecko")

    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    daily = df.groupby("date").agg(close=("close", "last")).reset_index()
    return daily


def fetch_fear_and_greed() -> pd.DataFrame:
    """Fetch the fear and greed index data."""

    response = requests.get(FEAR_GREED_API, params={"limit": 0, "format": "json"}, timeout=30)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise ValueError("No fear and greed data returned")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    df["fear_greed_index"] = pd.to_numeric(df["value"], errors="coerce")
    daily = df.groupby("date").agg(fear_greed_index=("fear_greed_index", "last")).reset_index()
    return daily


def filter_recent_years(price_df: pd.DataFrame, fng_df: pd.DataFrame, years: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter both dataframes to a trailing number of years based on price data."""

    cutoff = dt.date.today() - dt.timedelta(days=365 * years)
    return (
        price_df[price_df["date"] >= cutoff].reset_index(drop=True),
        fng_df[fng_df["date"] >= cutoff].reset_index(drop=True),
    )


def merged_market_and_sentiment(years: int = 7) -> pd.DataFrame:
    """Return merged price and sentiment data for the requested period."""

    prices = fetch_market_data()
    fng = fetch_fear_and_greed()
    prices, fng = filter_recent_years(prices, fng, years=years)
    merged = prices.merge(fng, on="date", how="left")
    merged.sort_values("date", inplace=True)
    return merged.reset_index(drop=True)


__all__ = [
    "fetch_market_data",
    "fetch_fear_and_greed",
    "filter_recent_years",
    "merged_market_and_sentiment",
]
