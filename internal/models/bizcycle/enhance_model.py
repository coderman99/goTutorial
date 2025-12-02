# enhance_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from hmmlearn.hmm import GaussianHMM



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
    # final fill
    X = X.ffill().bfill()

    return X

# ---- Hidden Markov Model helpers ----
def _build_state_label_map(model, labels, scaled_X):
    """
    Map each hidden state to the most frequent observed label in training.
    """

    states = model.predict(scaled_X)

    mapping = {}
    for state in np.unique(states):
        mask = states == state
        if mask.any():
            mapping[state] = labels.reset_index(drop=True)[mask].value_counts().idxmax()
    return mapping


def _fit_hmm(X, y):
    """Fit a Gaussian HMM with as many components as labels present."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_states = max(len(np.unique(y)), 2)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )
    model.fit(X_scaled)

    state_label_map = _build_state_label_map(model, y, X_scaled)

    label_encoder = LabelEncoder().fit(list(state_label_map.values()))

    return {
        "model": model,
        "scaler": scaler,
        "state_label_map": state_label_map,
        "label_encoder": label_encoder,
        "feature_order": list(X.columns),
    }
def train_hmm_multi_horizon(
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

    for h in horizons:
        y = df_targets[h].reindex(df_features.index).dropna()
        # align X to y
        X = df_features.reindex(y.index).copy()


        pipeline=_fit_hmm(X, y)
     
        model_path = os.path.join(model_dir, f"hmm_pipeline_{h}.joblib")
        joblib.dump(pipeline, model_path)
        pipelines[h] = pipeline
    return pipelines, None

# ---- Predict + explain function for a single latest row ----
def _summarize_feature_impacts(feature_names, state_means, top_n):
    """Return per-feature and aggregated indicator impacts."""
    # absolute magnitude for ranking
    impact_series = (
        pd.Series(state_means, index=feature_names)
        .abs()
        .sort_values(ascending=False)
    )

    top_features = []
    for feat in impact_series.head(top_n).index:
        idx = feature_names.tolist().index(feat)
        top_features.append({
            "feature": feat,
            "impact": float(state_means[idx]),
        })

    # aggregate by base indicator name (strip lag/suffixes)
    base_impacts = {}
    for feat, value in zip(feature_names, state_means):
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
def _compute_label_probabilities(state_probs, state_label_map):
    label_probs = {}
    for state, prob in enumerate(state_probs):
        label = state_label_map.get(state)
        if label is None:
            continue
        label_probs[label] = label_probs.get(label, 0.0) + float(prob)

    total = sum(label_probs.values())
    if total > 0:
        label_probs = {k: v / total for k, v in label_probs.items()}
    return label_probs
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

        # ---------------------------------------------------------
        # 1. Get expected feature order from scaler
        # ---------------------
        model = pipeline["model"]
        scaler = pipeline["scaler"]
        state_label_map = pipeline["state_label_map"]
        label_encoder = pipeline["label_encoder"]
        expected_cols = pipeline["feature_order"]

        # Reindex latest_row to match training order
        X = latest_row.reindex(columns=expected_cols)
        X = X.ffill(axis=1).bfill(axis=1).fillna(0)

      
        scaled = scaler.transform(X)
        logprob, state_probs = model.score_samples(scaled)
        state_probs = state_probs[0]
        pred_state = int(np.argmax(state_probs))
        pred_label = state_label_map.get(pred_state, label_encoder.classes_[0])
        label_probs = _compute_label_probabilities(state_probs, state_label_map)
        # ensure labels appear in encoder order for consistent output
        prob_dict = {label: float(label_probs.get(label, 0.0)) for label in label_encoder.classes_}
        # ---------------------------------------------------------
        # 4. Summaries
        # ---------------------------------------------------------
        state_means = model.means_[pred_state]
        
        top_features, top_indicators = _summarize_feature_impacts(
            expected_cols,state_means, top_n
        )

        results[horizon] = {
            "predicted_phase": pred_label,
            "probabilities": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
            "top_features": top_features,
            "most_impactful_indicators": top_indicators,
        }

    return results
