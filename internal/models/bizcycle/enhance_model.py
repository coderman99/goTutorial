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
def predict_and_explain(pipelines, X_all, top_n=10):
    """
    pipelines : dict of {horizon: (pipeline, label_encoder)}
    X_all     : full feature matrix used in training (must contain all engineered features)
    """

    latest_row = X_all.iloc[[-1]]       # (1 x n_features)
    results = {}

    for horizon, (pipe, le) in pipelines.items():

        # ---------------------------------------------------------
        # 1. Get expected feature order from scaler
        # ---------------------------------------------------------
        expected_cols = pipe.named_steps["scaler"].feature_names_in_

        # Reindex latest_row to match training order
        X = latest_row.reindex(columns=expected_cols)

        # ---------------------------------------------------------
        # 2. Predict probabilities and class label
        # ---------------------------------------------------------
        probs = pipe.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]

        # ---------------------------------------------------------
        # 3. SHAP explanation
        # ---------------------------------------------------------
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        scaled = pipe.named_steps["scaler"].transform(X)

        shap_raw = explainer.shap_values(scaled)

        # CASE A: List of arrays (old SHAP API, one per class)
        if isinstance(shap_raw, list):
            class_shap = shap_raw[pred_idx][0]   # shape (n_features,)

        # CASE B: SHAP Explanation object (new SHAP API)
        else:
            sv = explainer(X)
            # sv.values shape = (1, n_features, n_classes)
            class_shap = sv.values[0][:, pred_idx]

        # ---------------------------------------------------------
        # 4. Get top contributing features
        # ---------------------------------------------------------
        s = (
            pd.Series(class_shap, index=expected_cols)
              .abs()
              .sort_values(ascending=False)
        )

        top_list = [
            (feat, float(class_shap[expected_cols.tolist().index(feat)]))
            for feat in s.head(top_n).index
        ]

        results[horizon] = {
            "probs": dict(zip(le.inverse_transform(range(len(probs))), map(float, probs))),
            "pred": pred_label,
            "top_features": top_list
        }

    return results
