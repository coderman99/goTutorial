# enhance_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
import os

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - lightweight fallback
    from sklearn.ensemble import GradientBoostingClassifier

    _LGBM_AVAILABLE = False


if _LGBM_AVAILABLE:
    from lightgbm import early_stopping

    class LightGBMTimeSeriesClassifier(LGBMClassifier):
        """LightGBM classifier that keeps validation splits for early stopping."""

        def __init__(self, val_fraction=0.2, early_stopping_rounds=50, **kwargs):
            self.val_fraction = val_fraction
            self.early_stopping_rounds = early_stopping_rounds
            super().__init__(**kwargs)

        def fit(self, X, y, **kwargs):
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)
            X_train, y_train, X_val, y_val = _split_for_early_stopping(
                X_df, y_series, val_fraction=self.val_fraction
            )

            callbacks = kwargs.pop("callbacks", [])
            if X_val is not None and y_val is not None:
                callbacks.append(early_stopping(self.early_stopping_rounds, verbose=False))
                return super().fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="multi_logloss",
                    callbacks=callbacks,
                    **kwargs,
                )

            return super().fit(X_df, y_series, callbacks=callbacks, **kwargs)
else:
    LightGBMTimeSeriesClassifier = None


# ---- Helper: wide conversion if you still have long-format monthly data ----
def to_wide_monthly(df_long):
    """
    Accepts either:
      - long format df with index = timestamp AND column 'name'
      - OR wide format df (returns immediately)
    """

    df = df_long.copy()

    # ---- CASE 1: Already wide (no 'name' column) ----
    if "name" not in df.columns:
        return df

    # ---- CASE 2: Long format needs pivot ----
    # Ensure monthly timestamps in index
    if df.index.dtype != "datetime64[ns]":
        df.index = pd.to_datetime(df.index)

    df.index = df.index.to_period("M").to_timestamp("M")

    # Pivot to wide
    wide = df.pivot_table(
        index=df.index,
        columns="name",
        values="value",
        aggfunc="mean"
    )

    wide = wide.sort_index().ffill().bfill()

    return wide


# ---- Feature engineering ----
def make_features(wide, add_lags=(1,3,6), add_3m_smooth=True):
    """
    wide: DataFrame indexed by month with indicator columns
    returns: X (features DF)
    """
    X = wide.copy()
    # Remove or coerce non-numeric columns before creating percentage changes
    for col in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]):
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().any():
                X[col] = coerced
            else:
                X = X.drop(columns=col)

    # percent changes for level indicators can help
    # for price-like series: compute pct_change; for levels you may not want changepct, but keep generic
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    for col in numeric_cols:        
        X[f"{col}_pct"] = X[col].pct_change()

    # 3-month smoothed returns for SP500 if present (example column 'StockMarketIndex' or 'SP500' depending)
    sp_name_candidates = ['StockMarketIndex', 'SP500', 'SPX', 'sp500', 'StockMarketIndex']
    sp_col = None
    for p in sp_name_candidates:
        if p in X.columns:
            sp_col = p
            break
    if sp_col and add_3m_smooth:
        X['sp_3m_smooth'] = X[sp_col].pct_change(3)

    # Lag features
    for lag in add_lags:
        X_lag = X.shift(lag).add_suffix(f"_lag{lag}")
        X = pd.concat([X, X_lag], axis=1)

    # Drop rows with too many missing values
    X = X.dropna(thresh=int(X.shape[1]*0.5))

    # Remove columns with >50% NaN or constant
    X = X.loc[:, X.isna().mean() < 0.5]
    # Drop near-constant columns that add noise but little signal
    constant_mask = X.nunique(dropna=False) <= 1
    if constant_mask.any():
        X = X.loc[:, ~constant_mask]
    # final fill
    X = X.ffill().bfill()

    return X


def _screen_features_by_importance(X, y, max_features=150, min_score=0.0):
    """Select top predictive features using mutual information.

    The expanded Leading indicator set increases the number of columns and
    derived lags. To keep model training efficient and focused on informative
    signals, this helper keeps only the highest-scoring features.
    """

    if X.empty:
        return X, pd.Series(dtype=float)

    scores = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    score_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    selected = score_series[score_series > min_score].head(max_features).index
    if selected.empty:
        selected = score_series.head(max_features).index

    return X[selected], score_series

# ---- LightGBM helpers ----
def _split_for_early_stopping(X_df, y_series, val_fraction=0.2):
    """Return train/validation splits preserving time order for early stopping."""

    if len(X_df) < 5:
        return X_df, y_series, None, None

    split_idx = max(1, int(len(X_df) * (1 - val_fraction)))
    if split_idx >= len(X_df):
        split_idx = len(X_df) - 1

    X_train, X_val = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_train, y_val = y_series.iloc[:split_idx], y_series.iloc[split_idx:]
    return X_train, y_train, X_val, y_val


def _fit_lgbm(X, y):
    """Fit a LightGBM classifier with time-aware CV and early stopping."""

    # Ensure no missing values during training
    X_filled = X.ffill().bfill()

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    # Screen for the most informative features to reduce noise and training cost
    max_feats = max(20, min(200, X_filled.shape[1]))
    X_selected, feature_scores = _screen_features_by_importance(
        X_filled, y_encoded, max_features=max_feats, min_score=0.0
    )

    if _LGBM_AVAILABLE and len(y_encoded) > 2:
        base_estimator = LightGBMTimeSeriesClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            random_state=42,
            val_fraction=0.2,
            early_stopping_rounds=50,
            n_jobs=-1,
        )

        pipeline = Pipeline([
            ("clf", base_estimator),
        ])

        param_grid = {
            "clf__min_data_in_leaf": [20, 50],
            "clf__feature_fraction": [0.5, 0.8],
            "clf__bagging_fraction": [0.5, 0.8],
            "clf__lambda_l1": [0.1, 1],
            "clf__lambda_l2": [0.1, 1],
        }

        n_splits = max(2, min(5, len(y_encoded) - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_selected, y_encoded)
        best_model = grid_search.best_estimator_
        train_accuracy = float(grid_search.best_score_)
        model = best_model
    else:
        if _LGBM_AVAILABLE:
            model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                random_state=42,
            )
        else:
            model = GradientBoostingClassifier(random_state=42)

        model.fit(X_selected, y_encoded)
        y_pred_encoded = model.predict(X_selected)
        train_accuracy = accuracy_score(y_encoded, y_pred_encoded)

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_order": list(X_selected.columns),
        "feature_scores": feature_scores.to_dict(),
        "train_accuracy": float(train_accuracy),
    }
def train_lightgbm_multi_horizon(
    df_features,
    df_targets,
    horizons=['cycle_1m', 'cycle_3m', 'cycle_6m'],
    model_dir="models",
):
    """
    df_features: DataFrame indexed by month (features)
    df_targets: DataFrame with horizons columns aligned with df_features index
    Returns: dict of trained pipelines and dict of shap explainers
    """
    os.makedirs(model_dir, exist_ok=True)
    pipelines = {}
    accuracies = {}
    for h in horizons:
        y = df_targets[h].reindex(df_features.index).dropna()
        # align X to y
        X = df_features.reindex(y.index).copy()
        # If no labels are available for this horizon, skip but keep an explicit
        # None entry so downstream code can guard against missing accuracies.
        if y.empty:
            pipelines[h] = None
            accuracies[h] = None
            continue

        pipeline = _fit_lgbm(X, y)

        model_path = os.path.join(model_dir, f"lgbm_pipeline_{h}.joblib")
        joblib.dump(pipeline, model_path)
        pipelines[h] = pipeline
        accuracies[h] = pipeline["train_accuracy"]

    return pipelines, accuracies
# ---- Predict + explain function for a single latest row ----
def _summarize_feature_impacts(feature_names, importances, top_n):
    """Return per-feature and aggregated indicator impacts."""
    feature_list = list(feature_names)
    # absolute magnitude for ranking
    impact_series = (
        pd.Series(importances, index=feature_list)
        .abs()
        .sort_values(ascending=False)
    )
    top_features = [
        {"feature": feat, "impact": float(impact_series.loc[feat])}
        for feat in impact_series.head(top_n).index
    ]

    
    # aggregate by base indicator name (strip lag/suffixes)
    base_impacts = {}
    for feat, value in zip(feature_list, importances):
        base = feat
        if "_lag" in base:
            base = base.split("_lag")[0]
        if base.endswith("_pct"):
            base = base.replace("_pct", "")
        base_impacts.setdefault(base, 0.0)
        base_impacts[base] += abs(float(value))

    top_indicators = sorted(
        ({"indicator": k, "aggregate_impact": v} for k, v in base_impacts.items()),
        key=lambda x: x["aggregate_impact"],
        reverse=True,
    )[:top_n]

    return top_features, top_indicators


def _select_prediction_row(X_all, as_of=None):
    """Return a single-row DataFrame for the requested timestamp (or latest)."""

    if as_of is None:
        return X_all.iloc[[-1]]

    ts = pd.to_datetime(as_of)
    if ts in X_all.index:
        return X_all.loc[[ts]]

    # Allow monthly string such as "2024-03" that may map to end-of-month index
    monthly_ts = ts.to_period("M").to_timestamp("M")
    if monthly_ts in X_all.index:
        return X_all.loc[[monthly_ts]]

    raise KeyError(f"No feature row found for as_of={as_of}")

def predict_and_explain(pipelines, X_all, top_n=10, explainers=None, as_of=None):
    """
    Produce predictions for each horizon with probabilities and top drivers.

    Parameters
    ----------
    pipelines : dict
        Mapping of horizon → (trained pipeline, label encoder).
    X_all : DataFrame
        Full feature matrix used in training (must contain engineered columns).
    top_n : int
        Number of top features/indicators to include per horizon.
    explainers : dict, optional
        Optional mapping of horizon → explainer tuple from training to avoid
        rebuilding explainers during inference.
    """

    latest_row = _select_prediction_row(X_all, as_of)       # (1 x n_features)
    results = {
        "as_of": latest_row.index[-1].isoformat()
    }

    for horizon, pipeline in pipelines.items():
        if pipeline is None:
            results[horizon] = {
                "predicted_phase": None,
                "probabilities": {},
                "top_features": [],
                "most_impactful_indicators": [],
            }
            continue

        # ---------------------------------------------------------
        # 1. Get expected feature order from scaler
        # ---------------------
        model = pipeline["model"]
        estimator = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model

        label_encoder = pipeline["label_encoder"]
        expected_cols = pipeline["feature_order"]

        # Reindex latest_row to match training order
        X = latest_row.reindex(columns=expected_cols)
        X = X.ffill(axis=1).bfill(axis=1).fillna(0)

        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        prob_dict = {label: float(prob) for label, prob in zip(label_encoder.classes_, proba)}

        feature_importances = getattr(estimator, "feature_importances_", np.zeros(len(expected_cols)))

        
        top_features, top_indicators = _summarize_feature_impacts(
            expected_cols, feature_importances, top_n
        )

        results[horizon] = {
            "predicted_phase": pred_label,
            "probabilities": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
            "top_features": top_features,
            "most_impactful_indicators": top_indicators,
        }

    return results
