# enhance_model.py
import importlib
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from internal.models.bizcycle.indicator_definitions import KEY_INDICATORS

# Optional SHAP import: keep this isolated so missing extras never break parsing
shap = None
try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

# Prefer XGBoost when available, otherwise fall back gracefully
_XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback
    from sklearn.ensemble import GradientBoostingClassifier

    XGBClassifier = None


# ---- Helper: wide conversion for long-format indicator data ----
def _ensure_unique_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a unique, sorted index.

    When upstream joins accidentally introduce duplicate timestamps or
    MultiIndex rows, many pandas operations (for example ``reindex``) will
    raise a ``ValueError``.  This helper collapses duplicates deterministically
    using the last observed row so downstream feature engineering remains
    stable.
    """

    df = df.copy()

    if df.index.is_unique:
        return df.sort_index()

    if isinstance(df.index, pd.MultiIndex):
        # Preserve the most recent observation for each MultiIndex key
        df = df.groupby(level=list(range(df.index.nlevels))).last()
    else:
        df = df[~df.index.duplicated(keep="last")]

    return df.sort_index()


def to_wide_monthly(df_long, freq="M"):
    """
    Convert long-format indicators into a wide table without dropping duplicate
    timestamps. ``freq`` controls the aggregation endpoints.

    Accepts either:
      - long format df with index = timestamp AND column 'name'
      - OR wide format df (returns immediately)
    """
    df = df_long.copy()

    # Already wide (no 'name' column)
    if "name" not in df.columns:
        return df

    # Ensure timestamp exists only as a column to avoid index/column ambiguity.
    has_timestamp_column = "timestamp" in df.columns
    timestamp_in_index = "timestamp" in (df.index.names or [])

    if timestamp_in_index and has_timestamp_column:
        df = df.reset_index()
    elif timestamp_in_index and not has_timestamp_column:
        df = df.reset_index()
    elif not has_timestamp_column:
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "timestamp"})

    df.index.name = None

    # Normalize timestamp to requested end-of-period for consistent grouping
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].dt.to_period(freq).dt.to_timestamp(how="end")

    # Remove duplicate indicator rows; keep last observation in the period
    df = df.sort_values(["timestamp", "name"]).drop_duplicates(
        subset=["timestamp", "name"], keep="last"
    )

    wide = df.pivot_table(
        index="timestamp",
        columns="name",
        values="value",
        aggfunc="mean",
    )

    wide = _ensure_unique_sorted_index(wide).ffill()
    return wide


# ---- Feature engineering ----
def make_features(wide, base_lags=(1, 3)):
    """
    Build a compact, leakage-safe feature set.

    - Standardize macro indicators with Z-scores.
    - Remove stock-market return features to avoid leakage from SP500 labels.
    - Limit lagged indicators to trim the engineered feature count.
    """
    wide = _ensure_unique_sorted_index(wide)

    # Keep only numeric columns that are not targets/composite scores
    numeric_cols = [
        c
        for c in wide.columns
        if pd.api.types.is_numeric_dtype(wide[c])
        and not c.startswith("composite_cycle_")
        and not c.startswith("cycle_")
    ]
    X = wide[numeric_cols].copy()

    # Drop any obvious SPX/return columns defensively
    leak_cols = [c for c in X.columns if "return" in c.lower() or "sp500" in c.lower()]
    if leak_cols:
        X = X.drop(columns=leak_cols)

    # Rate-of-change (ROC) features capture short-term momentum in each indicator
    roc_windows = (1, 3, 6)
    roc_suffixes = tuple(f"roc{w}" for w in roc_windows)
    roc_frames = []
    for window in roc_windows:
        roc = X.pct_change(periods=window, fill_method=None)
        roc.columns = [f"{c}_roc{window}" for c in X.columns]
        roc_frames.append(roc)

    # Standardize each series (Z-score) to put them on comparable scales
    X = (X - X.mean()) / X.std(ddof=0)
    if roc_frames:
        standardized_roc = []
        for frame in roc_frames:
            frame_std = (frame - frame.mean()) / frame.std(ddof=0)
            standardized_roc.append(frame_std)
        X = pd.concat([X] + standardized_roc, axis=1)

    # Compress extremely wide indicator sets to a smaller latent representation
    base_indicator_cols = [c for c in X.columns if not c.endswith(roc_suffixes)]
    if len(base_indicator_cols) > 100:
        compressed = _reduce_dimensionality(
            X[base_indicator_cols],
            target_components=min(64, len(base_indicator_cols) // 2),
        )
        X = pd.concat([compressed, X.drop(columns=base_indicator_cols)], axis=1)

    # Add a few simple lags for each feature
    lagged_frames = []
    for lag in base_lags:
        lagged = X.shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in X.columns]
        lagged_frames.append(lagged)

    if lagged_frames:
        X = pd.concat([X] + lagged_frames, axis=1)

    X = X.dropna(how="all")
    return X


def _reduce_dimensionality(X_df, target_components=50):
    """Compress high-dimensional indicator sets using PCA with optional autoencoder boost."""

    X_filled = X_df.ffill().bfill()
    X_filled = X_filled.dropna(axis=0, how="any")

    if X_filled.empty:
        return X_df

    target_components = max(1, min(target_components, X_filled.shape[1]))

    autoencoded = _autoencode_features(X_filled, bottleneck=target_components)
    if autoencoded is not None:
        return autoencoded

    n_components = min(target_components, X_filled.shape[0], X_filled.shape[1])
    if n_components < 1:
        return X_df

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X_filled)
    component_names = [f"pca_component_{i+1}" for i in range(transformed.shape[1])]
    return pd.DataFrame(transformed, index=X_filled.index, columns=component_names)


def _autoencode_features(X_df, bottleneck=32, epochs=20, batch_size=32, random_state=42):
    """Try to learn a compact latent representation using a lightweight autoencoder.

    Returns a DataFrame of encoded features, or ``None`` when TensorFlow is not available.
    """

    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        return None

    tensorflow = importlib.import_module("tensorflow")
    keras = tensorflow.keras

    tensorflow.random.set_seed(random_state)

    input_dim = X_df.shape[1]
    bottleneck = max(1, min(bottleneck, input_dim))
    hidden_dim = max(bottleneck * 2, bottleneck + 8)

    inputs = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    bottleneck_layer = keras.layers.Dense(bottleneck, activation="linear", name="bottleneck")(encoded)
    decoded = keras.layers.Dense(hidden_dim, activation="relu")(bottleneck_layer)
    outputs = keras.layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_df.values,
        X_df.values,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    encoder = keras.Model(inputs, bottleneck_layer)
    encoded_features = encoder.predict(X_df.values, verbose=0)
    columns = [f"ae_component_{i+1}" for i in range(encoded_features.shape[1])]
    return pd.DataFrame(encoded_features, index=X_df.index, columns=columns)


def _screen_features_by_importance(X, y, max_features=60, min_score=0.0):
    """
    Screen features using mutual information.

    To keep model training efficient and focused on informative
    signals, this helper keeps only the highest-scoring features.
    """
    if X.empty:
        return X, pd.Series(dtype=float)

    scores = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    score_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    selected = score_series[score_series > min_score].head(max_features).index
    if selected.empty:
        selected = score_series.head(max_features).index

    # Preserve diversity by forcing inclusion of key macro indicators when present
    key_features_in_X = [col for col in X.columns if any(col.startswith(k) for k in KEY_INDICATORS)]
    selected = pd.Index(selected).union(key_features_in_X)

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

    # Class balancing for imbalanced macro cycles
    class_counts = pd.Series(y_encoded).value_counts()
    total = class_counts.sum()
    class_weight = {cls: total / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}
    train_sample_weight = pd.Series(y_train).map(class_weight).to_numpy()

    if _XGBOOST_AVAILABLE:
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob" if len(class_counts) > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if len(class_counts) > 2 else "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        if len(class_counts) == 2 and class_counts.min() > 0:
            xgb_params["scale_pos_weight"] = float(class_counts.max() / class_counts.min())

        model = XGBClassifier(**xgb_params)

        if X_val is not None and y_val is not None:
            eval_sample_weight = pd.Series(y_val).map(class_weight).to_numpy()
            model.fit(
                X_train,
                y_train,
                sample_weight=train_sample_weight,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[eval_sample_weight],
                verbose=False,
            )
        else:
            full_sample_weight = pd.Series(y_encoded).map(class_weight).to_numpy()
            model.fit(X_selected, y_encoded, sample_weight=full_sample_weight, verbose=False)
    else:  # pragma: no cover - fallback for environments without XGBoost
        from sklearn.ensemble import GradientBoostingClassifier

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


# Backwards-compatible alias
_fit_lightgbm = _fit_xgboost


def train_xgboost_multi_horizon(
    df_features,
    df_targets,
    horizons=("cycle_1m", "cycle_3m", "cycle_6m"),
    model_dir="models",
):
    """
    df_features: DataFrame indexed by time (features)
    df_targets: DataFrame with horizons columns aligned with df_features index
    Returns: dict of trained pipelines and dict of train accuracies
    """
    os.makedirs(model_dir, exist_ok=True)
    pipelines = {}
    accuracies = {}

    for h in horizons:
        y = df_targets[h].reindex(df_features.index).dropna()
        X = df_features.reindex(y.index).copy()

        if y.empty:
            pipelines[h] = None
            accuracies[h] = None
            continue

        pipeline = _fit_lightgbm(X, y)

        model_path = os.path.join(model_dir, f"lightgbm_pipeline_{h}.joblib")
        joblib.dump(pipeline, model_path)
        pipelines[h] = pipeline
        accuracies[h] = pipeline["train_accuracy"]

    return pipelines, accuracies


# ---- Predict + explain for latest row ----
def _summarize_feature_impacts(feature_names, importances, top_n):
    """Return per-feature and aggregated indicator impacts."""
    feature_list = list(feature_names)
    impacts_array = np.asarray(importances)

    if impacts_array.ndim != 1 or impacts_array.shape[0] != len(feature_list):
        impacts_array = impacts_array.reshape(-1)[: len(feature_list)]

    impact_pairs = sorted(
        zip(feature_list, impacts_array),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_n]

    top_features = [
        {"feature": name, "impact": float(score)} for name, score in impact_pairs
    ]

    # Aggregate by indicator prefix before lag suffix
    indicator_impacts = {}
    for name, score in impact_pairs:
        base = name.split("_lag")[0]
        indicator_impacts[base] = indicator_impacts.get(base, 0.0) + float(score)

    top_indicators = sorted(
        [{"indicator": k, "impact": v} for k, v in indicator_impacts.items()],
        key=lambda d: abs(d["impact"]),
        reverse=True,
    )

    return top_features, top_indicators


def predict_and_explain(pipelines, X_latest, top_n=12):
    """
    Generate business-cycle predictions and feature explanations for the latest row.
    """
    if X_latest.empty:
        raise ValueError("No feature rows available for prediction")

    last_row = X_latest.iloc[[-1]]
    results = {}

    for horizon, pipeline in pipelines.items():
        if pipeline is None:
            continue

        model = pipeline["model"]
        label_encoder = pipeline["label_encoder"]
        expected_cols = pipeline["feature_order"]

        X = last_row.reindex(columns=expected_cols)
        X_filled, encoders = _encode_categoricals(X, pipeline.get("feature_encoders", {}))

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_filled)[0]
        else:
            raw_pred = model.predict(X_filled)
            proba = np.zeros(len(label_encoder.classes_))
            proba[int(raw_pred[0])] = 1.0

        pred_idx = int(np.argmax(proba))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        prob_dict = {label: float(p) for label, p in zip(label_encoder.classes_, proba)}

        booster = getattr(model, "booster_", None)
        if booster is not None:
            gain_importance = booster.feature_importance(importance_type="gain")
            feature_names = booster.feature_name()
            importance_map = {name: score for name, score in zip(feature_names, gain_importance)}
            feature_importances = [importance_map.get(col, 0.0) for col in expected_cols]
        else:
            feature_importances = getattr(
                model, "feature_importances_", np.zeros(len(expected_cols))
            )

        # Prefer SHAP explanations when available
        shap_impacts = None
        if shap is not None and hasattr(model, "predict"):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_filled)
                if isinstance(shap_values, list):
                    shap_array = np.asarray(shap_values[pred_idx][0])
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        shap_array = shap_values[0, pred_idx, :]
                    elif shap_values.ndim == 2:
                        shap_array = shap_values[0]
                    else:
                        shap_array = shap_values.squeeze()
                else:
                    shap_array = None
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


# Backwards-compatible alias for earlier name
train_lightgbm_multi_horizon = train_xgboost_multi_horizon
