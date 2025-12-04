import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA, Bollinger Bands, and Stochastic RSI.

    Expects columns: [date, close]. Returns dataframe with added columns:
    ema_20, bb_middle, bb_upper, bb_lower, stoch_rsi.
    """

    result = df.copy()
    result.sort_values("date", inplace=True)

    close = result["close"]

    # Exponential Moving Average
    result["ema_20"] = close.ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    result["bb_middle"] = rolling_mean
    result["bb_upper"] = rolling_mean + 2 * rolling_std
    result["bb_lower"] = rolling_mean - 2 * rolling_std

    # RSI calculation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    result["stoch_rsi"] = (rsi - rsi_min) / (rsi_max - rsi_min)

    return result


__all__ = ["compute_indicators"]
