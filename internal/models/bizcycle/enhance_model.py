# enhance_model.py
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - lightweight fallback
    from sklearn.ensemble import GradientBoostingClassifier

    _XGBOOST_AVAILABLE = False


# ---- Helper: wide conversion if you still have long-format monthly data ----
def to_wide_monthly(df_long):
    """
    Convert long-format monthly indicators into a wide table without dropping
    duplicate timestamps.

    Accepts either:
      - long format df with index = timestamp AND column 'name'
      - OR wide format df (returns immediately)
    """

    df = df_long.copy()

    # ---- CASE 1: Already wide (no 'name' column) ----
    if "name" not in df.columns:
        return df

    # ---- CASE 2: Long format needs pivot ----
    # Ensure timestamp exists only as a column to avoid index/column ambiguity.
    has_timestamp_column = "timestamp" in df.columns
    timestamp_in_index = "timestamp" in (df.index.names or [])

    if timestamp_in_index and has_timestamp_column:
        # Drop the index level and keep the explicit column
        df = df.reset_index()
    elif timestamp_in_index and not has_timestamp_column:
        df = df.reset_index()
    elif not has_timestamp_column:
        # No column but also no named index: create one from the index values
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "timestamp"})

    # Ensure index is unnamed to avoid accidental clashes after reset
    df.index.name = None

    # Normalize timestamp to month-end for consistent grouping
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].dt.to_period("M").dt.to_timestamp("M")

    # Remove duplicate month/indicator rows that can appear when historical
    # files are appended multiple times; keep the most recent observation.
    df = df.sort_values(["timestamp", "name"]).drop_duplicates(
        subset=["timestamp", "name"], keep="last"
    )

    # Pivot to wide with aggregation (mean keeps all rows for the month)
    wide = df.pivot_table(
        index="timestamp",
        columns="name",
        values="value",
        aggfunc="mean",
    )

    wide = wide.sort_index().ffill()

    return wide


# ---- Feature engineering ----
def make_features(wide, base_lags=(1, 3), return_lags=(1, 3, 6)):
    """
    Build a compact, leakage-safe feature set.

    - Standardize macro indicators with Z-scores.
    - Keep only raw returns plus a few lags (no rolling/pct noise).
    - Limit lagged indicators to trim the engineered feature count.
    """

    wide = wide.copy()
    wide = wide.loc[~wide.index.duplicated()].sort_index()

    # Keep only numeric columns
    numeric_cols = [c for c in wide.columns if pd.api.types.is_numeric_dtype(wide[c])]
    X = wide[numeric_cols].copy()

    # Standardize each indicator so scales are comparable
    for col in list(X.columns):
        std = X[col].std()
        if std is None or std == 0 or pd.isna(std):
            # Drop constant/non-informative columns early
            X = X.drop(columns=col)
            continue
        mean = X[col].mean()
        X[col] = (X[col] - mean) / std

    features = pd.DataFrame(index=X.index)

    # Preserve standardized indicators (excluding returns which get special handling)
    return_col = "returns" if "returns" in X.columns else None
    indicator_cols = [c for c in X.columns if c != return_col]
    features = pd.concat([features, X[indicator_cols]], axis=1)

    for lag in base_lags:
        lagged = X[indicator_cols].shift(lag).add_suffix(f"_lag{lag}")
        features = pd.concat([features, lagged], axis=1)

    # Add return features only after resampled alignment
    if return_col:
        features["returns"] = wide[return_col]
        for lag in return_lags:
            features[f"returns_lag{lag}"] = wide[return_col].shift(lag)

    features = features.dropna(how="all")
    return features


def _screen_features_by_importance(X, y, max_features=60, min_score=0.0):
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


def _encode_categoricals(X_df, fitted_encoders=None):
    """Encode categorical columns with LabelEncoder, returning encoded frame + encoders."""

    encoders = fitted_encoders or {}
    X_encoded = X_df.copy()

    cat_cols = [
        c
        for c in X_encoded.columns
        if X_encoded[c].dtype == "object" or pd.api.types.is_categorical_dtype(X_encoded[c])
    ]

    for col in cat_cols:
        enc = encoders.get(col)
        if enc is None:
            enc = LabelEncoder()
            enc.fit(X_encoded[col].astype(str).fillna("<NA>"))
            encoders[col] = enc

        X_encoded[col] = enc.transform(X_encoded[col].astype(str).fillna("<NA>"))

    return X_encoded, encoders

# ---- CatBoost helpers ----
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


def _fit_xgboost(X, y):
    """Fit an XGBoost classifier with time-aware validation for early stopping."""

    # Encode categorical features before any filling/selection
    X_encoded, feature_encoders = _encode_categoricals(X)

    # Forward-fill to avoid peeking into the future, then drop any rows that
    # still contain gaps so mutual_info and training do not receive NaNs.
    X_filled = X_encoded.ffill()
    if isinstance(y, pd.Series):
        y_aligned = y.copy()
    else:
        y_aligned = pd.Series(y, index=X_filled.index)

    valid_mask = X_filled.notna().all(axis=1)
    X_filled = X_filled.loc[valid_mask]
    y_aligned = y_aligned.loc[valid_mask]

    if X_filled.empty or y_aligned.empty:
        raise ValueError("No valid samples after dropping rows with missing features")

    label_encoder = LabelEncoder().fit(y_aligned)
    y_encoded = label_encoder.transform(y_aligned)

    max_feats = max(20, min(60, X_filled.shape[1]))
    X_selected, feature_scores = _screen_features_by_importance(
        X_filled, y_encoded, max_features=max_feats, min_score=0.0
    )

    X_train, y_train, X_val, y_val = _split_for_early_stopping(
        X_selected, pd.Series(y_encoded, index=X_selected.index), val_fraction=0.2
    )

    if _XGBOOST_AVAILABLE:
        # Class balancing for imbalanced macro cycles
        class_counts = pd.Series(y_encoded).value_counts()
        total = class_counts.sum()
        class_weight = {cls: total / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}
        sample_weight = pd.Series(y_train).map(class_weight).to_numpy()

        # scale_pos_weight is primarily for binary tasks; we provide a balanced
        # ratio here to honor class imbalance without changing the multi-class objective.
        imbalance_ratio = float(class_counts.max() / class_counts.min()) if len(class_counts) > 1 else 1.0

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            objective="multi:softprob",
            random_state=42,
            scale_pos_weight=imbalance_ratio,
            tree_method="hist",
            verbosity=0,
        )

        if X_val is not None and y_val is not None:
            eval_sample_weight = pd.Series(y_val).map(class_weight).to_numpy()
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[eval_sample_weight],
                verbose=False,
            )
        else:
            model.fit(X_selected, y_encoded, sample_weight=pd.Series(y_encoded).map(class_weight))
    else:  # pragma: no cover - fallback for environments without xgboost
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_selected, y_encoded)

    y_pred_encoded = model.predict(X_selected)
    train_accuracy = accuracy_score(y_encoded, y_pred_encoded)

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_order": list(X_selected.columns),
        "feature_scores": feature_scores.to_dict(),
        "feature_encoders": {k: v.classes_.tolist() for k, v in feature_encoders.items()},
        "train_accuracy": float(train_accuracy),
    }


def train_xgboost_multi_horizon(
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

        pipeline = _fit_xgboost(X, y)

        model_path = os.path.join(model_dir, f"xgboost_pipeline_{h}.joblib")
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
        fitted_feature_encs = pipeline.get("feature_encoders", {})

        # Reindex latest_row to match training order and encode categoricals consistently
        X = latest_row.reindex(columns=expected_cols)
        X = X.ffill(axis=1).bfill(axis=1).fillna(0)

        # Rehydrate encoders
        encoders = {}
        for col, classes in fitted_feature_encs.items():
            enc = LabelEncoder()
            enc.classes_ = np.array(classes)
            encoders[col] = enc

        X_encoded, _ = _encode_categoricals(X, encoders)

        raw_pred = estimator.predict(X_encoded)
        if raw_pred.ndim == 2:  # XGBoost with multi:softprob returns probability matrix
            proba = raw_pred[0]
        elif hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_encoded)[0]
        else:
            proba = np.zeros(len(label_encoder.classes_))
            proba[int(raw_pred[0])] = 1.0

        pred_idx = int(np.argmax(proba))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        prob_dict = {label: float(prob) for label, prob in zip(label_encoder.classes_, proba)}

        booster = getattr(estimator, "get_booster", lambda: None)()
        if booster is not None:
            raw_importance = booster.get_score(importance_type="gain")
            if raw_importance and all(k.startswith("f") for k in raw_importance):
                feature_importances = [raw_importance.get(f"f{i}", 0.0) for i in range(len(expected_cols))]
            else:
                feature_importances = [raw_importance.get(col, 0.0) for col in expected_cols]
        else:
            feature_importances = getattr(estimator, "feature_importances_", np.zeros(len(expected_cols)))

        # Prefer SHAP explanations when available
        shap_impacts = None
        if shap is not None and hasattr(estimator, "predict"):
            try:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_encoded)
                if isinstance(shap_values, list):
                    shap_array = shap_values[pred_idx][0]
                else:
                    shap_array = shap_values[0]
                shap_impacts = shap_array
            except Exception:
                shap_impacts = None

        impacts_for_ranking = shap_impacts if shap_impacts is not None else feature_importances

        top_features, top_indicators = _summarize_feature_impacts(
            expected_cols, impacts_for_ranking, top_n
        )

        results[horizon] = {
            "predicted_phase": pred_label,
            "probabilities": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
            "top_features": top_features,
            "most_impactful_indicators": top_indicators,
        }

    return results


# Backwards-compatible aliases for callers expecting earlier names
train_catboost_multi_horizon = train_xgboost_multi_horizon
train_lightgbm_multi_horizon = train_xgboost_multi_horizon
