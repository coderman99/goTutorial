# enhance_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap
import os

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

    df.index = df.index.to_period("M").to_timestamp()

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
    # percent changes for level indicators can help
    # for price-like series: compute pct_change; for levels you may not want changepct, but keep generic
    for col in X.columns:
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

# ---- Train one model per horizon with time-aware CV & return trained model + scaler + label encoder ----
def train_xgb_multi_horizon(df_features, df_targets, horizons=['cycle_1m','cycle_3m','cycle_6m'],
                            model_dir="models", n_splits=5):
    """
    df_features: DataFrame indexed by month (features)
    df_targets: DataFrame with horizons columns aligned with df_features index
    Returns: dict of trained pipelines and dict of shap explainers
    """
    os.makedirs(model_dir, exist_ok=True)
    pipelines = {}
    explainers = {}

    tss = TimeSeriesSplit(n_splits=n_splits)

    for h in horizons:
        y = df_targets[h].reindex(df_features.index).dropna()
        # align X to y
        X = df_features.reindex(y.index).copy()

        # encode target classes to integers
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # pipeline: scaler -> XGB
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=4
            ))
        ])

        # small grid to avoid long runs — tune if you have time
        param_grid = {
            'clf__n_estimators': [100, 300],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.01, 0.1],
        }

        g = GridSearchCV(pipe, param_grid, cv=tss, scoring='accuracy', n_jobs=1, verbose=1)
        g.fit(X, y_enc)

        print(f"Best params for {h}: {g.best_params_}, best_score={g.best_score_:.3f}")

        # save pipeline + label encoder
        model_path = os.path.join(model_dir, f"pipeline_{h}.joblib")
        le_path = os.path.join(model_dir, f"labelenc_{h}.joblib")
        joblib.dump(g.best_estimator_, model_path)
        joblib.dump(le, le_path)
        pipelines[h] = (g.best_estimator_, le)

        # SHAP explainer on a sample (explain the whole training set can be slow)
        # We need raw model and scaled X to compute shap values; use scaled features
        scaler = g.best_estimator_.named_steps['scaler']
        clf = g.best_estimator_.named_steps['clf']
        Xs = scaler.transform(X)
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(Xs)  # list per class
        explainers[h] = (explainer, shap_vals, X.index, X.columns)

        # save explainer (optional)
        joblib.dump(explainer, os.path.join(model_dir, f"explainer_{h}.joblib"))

    return pipelines, explainers

# ---- Predict + explain function for a single latest row ----
def _summarize_feature_impacts(feature_names, shap_values, top_n):
    """Return per-feature and aggregated indicator impacts."""
    # absolute magnitude for ranking
    impact_series = (
        pd.Series(shap_values, index=feature_names)
        .abs()
        .sort_values(ascending=False)
    )

    top_features = []
    for feat in impact_series.head(top_n).index:
        idx = feature_names.tolist().index(feat)
        top_features.append({
            "feature": feat,
            "impact": float(shap_values[idx]),
        })

    # aggregate by base indicator name (strip lag/suffixes)
    base_impacts = {}
    for feat, value in zip(feature_names, shap_values):
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


def _resolve_explainer(explainers, horizon, pipe):
    """Return a cached explainer when available, else build a fresh TreeExplainer."""

    if explainers and horizon in explainers:
        entry = explainers[horizon]
        # train_xgb_multi_horizon stores (explainer, shap_vals, index, cols)
        if isinstance(entry, tuple) and len(entry) and hasattr(entry[0], "shap_values"):
            return entry[0]
        if hasattr(entry, "shap_values"):
            return entry
    return shap.TreeExplainer(pipe.named_steps["clf"])


def _extract_shap_for_prediction(explainer, scaled_X, raw_X, pred_idx):
    """Handle SHAP APIs for a single-row prediction safely."""
    shap_raw = explainer.shap_values(scaled_X)

    # CASE A: List of arrays (old SHAP API, one per class)
    if isinstance(shap_raw, list):
        return shap_raw[pred_idx][0]   # shape (n_features,)

    # CASE B: SHAP Explanation object (new SHAP API)
    sv = explainer(raw_X)
    # sv.values shape = (1, n_features, n_classes)
    return sv.values[0][:, pred_idx]


def _select_prediction_row(X_all, as_of=None):
    """Return a single-row DataFrame for the requested timestamp (or latest)."""

    if as_of is None:
        return X_all.iloc[[-1]]

    ts = pd.to_datetime(as_of)
    if ts in X_all.index:
        return X_all.loc[[ts]]

    # Allow monthly string such as "2024-03" that may map to end-of-month index
    monthly_ts = ts.to_period("M").to_timestamp()
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

    for horizon, (pipe, le) in pipelines.items():

        # ---------------------------------------------------------
        # 1. Get expected feature order from scaler
        # ---------------------------------------------------------
        expected_cols = pipe.named_steps["scaler"].feature_names_in_

        # Reindex latest_row to match training order
        X = latest_row.reindex(columns=expected_cols)
        X = X.ffill(axis=1).bfill(axis=1).fillna(0)

        # ---------------------------------------------------------
        # 2. Predict probabilities and class label
        # ---------------------------------------------------------
        probs = pipe.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]

        prob_dict = dict(
            zip(le.inverse_transform(range(len(probs))), map(float, probs))
        )

        # ---------------------------------------------------------
        # 3. SHAP explanation (reuse cached explainer when provided)
        # ---------------------------------------------------------
        explainer = _resolve_explainer(explainers, horizon, pipe)
        scaled = pipe.named_steps["scaler"].transform(X)

        class_shap = _extract_shap_for_prediction(explainer, scaled, X, pred_idx)

        # ---------------------------------------------------------
        # 4. Summaries
        # ---------------------------------------------------------
        top_features, top_indicators = _summarize_feature_impacts(
            expected_cols, class_shap, top_n
        )

        results[horizon] = {
            "predicted_phase": pred_label,
            "probabilities": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
            "top_features": top_features,
            "most_impactful_indicators": top_indicators,
        }

    return results
