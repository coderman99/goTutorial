"""Model training utilities for Bitcoin trading signals.

This module derives buy/hold/sell labels from the indicator dataset, trains
classification and regression models, and offers a helper for producing the
latest signal with an expected holding horizon.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .main import build_dataset


FEATURE_COLUMNS = [
    "close",
    "ema_20",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "stoch_rsi",
    "fear_greed_index",
]

ACTION_MAP: Dict[int, str] = {1: "buy", 0: "hold", -1: "sell"}


@dataclass
class ModelBundle:
    """Container for the trained models and their metrics."""

    action_model: Pipeline
    hold_model: RandomForestRegressor
    feature_columns: Tuple[str, ...]
    reports: Dict[str, str]


def _label_actions(
    df: pd.DataFrame, horizon: int = 14, buy_threshold: float = 0.03, sell_threshold: float = -0.03
) -> pd.DataFrame:
    """Assign buy/hold/sell labels based on a forward return window."""

    labeled = df.copy()
    labeled.sort_values("date", inplace=True)

    forward_return = labeled["close"].shift(-horizon) / labeled["close"] - 1
    labeled["action"] = np.select(
        [forward_return >= buy_threshold, forward_return <= sell_threshold], [1, -1], default=0
    )

    # Drop rows without enough lookahead data
    return labeled.iloc[:-horizon].copy()


def _add_hold_days(df: pd.DataFrame, exit_threshold: float = 0.04, max_horizon: int = 45) -> pd.DataFrame:
    """Compute holding horizon until an exit threshold is reached or max horizon elapses."""

    augmented = df.copy()
    closes = augmented["close"].to_numpy()
    hold_days = []

    for idx, current_price in enumerate(closes):
        days = max_horizon
        for offset in range(1, max_horizon + 1):
            if idx + offset >= len(closes):
                break
            change = closes[idx + offset] / current_price - 1
            if abs(change) >= exit_threshold:
                days = offset
                break
        hold_days.append(days)

    augmented["hold_days"] = hold_days
    return augmented


def prepare_training_data(
    horizon: int = 14,
    buy_threshold: float = 0.03,
    sell_threshold: float = -0.03,
    exit_threshold: float = 0.04,
    max_horizon: int = 45,
) -> pd.DataFrame:
    """Fetch data, build indicators, and produce labeled rows for model training."""

    base_df = build_dataset()
    labeled = _label_actions(base_df, horizon=horizon, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    labeled = _add_hold_days(labeled, exit_threshold=exit_threshold, max_horizon=max_horizon)
    labeled.dropna(subset=FEATURE_COLUMNS + ["action", "hold_days"], inplace=True)
    return labeled


def train_models(
    horizon: int = 14,
    buy_threshold: float = 0.03,
    sell_threshold: float = -0.03,
    exit_threshold: float = 0.04,
    max_horizon: int = 45,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelBundle:
    """Train classification/regression models for trading signals."""

    data = prepare_training_data(
        horizon=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        exit_threshold=exit_threshold,
        max_horizon=max_horizon,
    )

    X = data[FEATURE_COLUMNS]
    y_action = data["action"]
    y_hold = data["hold_days"]

    X_train, X_val, y_action_train, y_action_val, y_hold_train, y_hold_val = train_test_split(
        X, y_action, y_hold, test_size=test_size, random_state=random_state, shuffle=True
    )

    action_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    action_model.fit(X_train, y_action_train)

    hold_model = RandomForestRegressor(
        n_estimators=300, random_state=random_state, min_samples_leaf=2, n_jobs=-1
    )
    hold_model.fit(X_train, y_hold_train)

    reports = {
        "classification": classification_report(y_action_val, action_model.predict(X_val), digits=3),
        "hold_mae": f"MAE: {mean_absolute_error(y_hold_val, hold_model.predict(X_val)):.2f} days",
    }

    return ModelBundle(
        action_model=action_model,
        hold_model=hold_model,
        feature_columns=tuple(FEATURE_COLUMNS),
        reports=reports,
    )


def predict_signal(bundle: ModelBundle, df: pd.DataFrame) -> Dict[str, float]:
    """Generate the latest trading signal and expected holding period."""

    latest = df.sort_values("date").iloc[[-1]]
    features = latest[list(bundle.feature_columns)]
    action_code = int(bundle.action_model.predict(features)[0])
    hold_days = float(bundle.hold_model.predict(features)[0])

    return {"action": ACTION_MAP.get(action_code, "hold"), "expected_hold_days": max(1.0, round(hold_days, 2))}


def train_and_predict() -> Tuple[ModelBundle, Dict[str, float]]:
    """Convenience wrapper to train models and emit the current signal."""

    bundle = train_models()
    dataset = build_dataset()
    dataset.dropna(subset=bundle.feature_columns, inplace=True)
    signal = predict_signal(bundle, dataset)
    return bundle, signal


__all__ = [
    "ModelBundle",
    "prepare_training_data",
    "train_models",
    "predict_signal",
    "train_and_predict",
]
