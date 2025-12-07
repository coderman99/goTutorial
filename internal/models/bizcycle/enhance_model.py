import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# CATEGORY COMPOSITE BUILDERS
# ------------------------------------------------------------

CATEGORY_MAP = {
    "Leading": "leading_comp",
    "Lagging": "lagging_comp",
    "Coincident": "coincident_comp",
}

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
    Build category-level composite indicators.
    Safe against multi-index issues.
    """

    if "indicator_cat" not in long_df.columns:
        print("[Composite] No indicator_cat column found — skipping composites.")
        return None

    comps = {}
    long_df = long_df.copy()

    # Ensure index is a clean timestamp index
    if isinstance(long_df.index, pd.MultiIndex):
        # Take only the timestamp level
        long_df.index = long_df.index.get_level_values(0)

    long_df.index = pd.to_datetime(long_df.index)
    long_df.index.name = "timestamp"

    categories = long_df["indicator_cat"].dropna().unique()

    for cat in categories:
        subset = long_df[long_df["indicator_cat"] == cat]

        if subset.empty:
            continue

    wide = _ensure_unique_sorted_index(wide).ffill()
    return wide






# ------------------------------------------------------------
# VOLATILITY MEASURES
# ------------------------------------------------------------
def add_category_volatility(comp_df):
    df = comp_df.copy()
    windows = [6, 12]

    for col in comp_df.columns:
        for win in windows:
            df[f"vol_{col}_win{win}"] = comp_df[col].rolling(win).std()

    # ensure no index drift
    df = df.reindex(df.index.unique().sort_values())

    return df




# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

def make_roc_features(df: pd.DataFrame, windows=[1, 3, 6]):
    df = df.copy()
    for win in windows:
        roc = df.pct_change(periods=win)
        roc = roc.replace([np.inf, -np.inf], np.nan)
        df[f"value_roc{win}"] = roc["value"]
    return df


def add_lagged_features(df: pd.DataFrame, lags=[1, 3]):
    df = df.copy()
    for lag in lags:
        df[f"value_lag{lag}"] = df["value"].shift(lag)
        for roc in [1, 3, 6]:
            col = f"value_roc{roc}"
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def drop_constant_columns(df: pd.DataFrame):
    df = df.copy()
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=const_cols, errors="ignore")
    return df, const_cols


# ------------------------------------------------------------
# MAIN FEATURE MATRIX CREATOR
# ------------------------------------------------------------

def make_features(long_df, wide_df):
    """
    Build the combined feature matrix.
    Fully index-aligned, composite-safe and volatility-safe.
    """
    wide = _ensure_unique_sorted_index(wide)

    # --------------------------
    # STEP 1 — Base features
    # --------------------------
    X = long_df[["value"]].copy()
    X = X.sort_index()

    # Ensure consistent index name
    if X.index.name is None:
        X.index.name = "timestamp"

    # Rate-of-change (ROC) features capture short-term momentum in each indicator
    roc_windows = (1, 3, 6)
    roc_suffixes = tuple(f"roc{w}" for w in roc_windows)
    roc_frames = []
    for window in roc_windows:
        roc = X.pct_change(periods=window, fill_method=None)
        roc.columns = [f"{c}_roc{window}" for c in X.columns]
        roc_frames.append(roc)

    # Add rate of change (ROC)
    for win in [1, 3, 6]:
        X[f"value_roc{win}"] = X["value"].pct_change(win)

    print(f"[Info] Base feature columns: {list(X.columns)}")

    # --------------------------
    # STEP 2 — Build composites
    # --------------------------
    print("[Stage] Computing category composites...")
    comp = build_category_composites(long_df)

    if comp is None or comp.empty:
        print("[Warning] No composites created. Continuing without category features.")
    else:
        print(f"[Success] Composite columns created: {list(comp.columns)}")

        # Ensure composite index matches feature index
        comp.index = pd.to_datetime(comp.index)
        comp = comp.reindex(X.index)  # align 1:1 with X

        # --------------------------
        # STEP 3 — Add volatility
        # --------------------------
        comp = add_category_volatility(comp)

        # Ensure again that indexes align
        comp.index = comp.index.rename(X.index.name)
        comp = comp.reindex(X.index)

        print(f"[Success] After volatility, columns: {list(comp.columns)}")

        # --------------------------
        # JOIN COMPOSITES WITH X
        # --------------------------
        X = X.join(comp, how="left")

    # --------------------------
    # STEP 4 — Add wide monthly dataframe (SP500 etc)
    # --------------------------
    if wide_df is not None and not wide_df.empty:
        wide_df.index = pd.to_datetime(wide_df.index)
        wide_df.index = wide_df.index.rename(X.index.name)
        wide_df = wide_df.reindex(X.index)

        print(f"[Stage] Joining wide monthly features: {wide_df.columns.tolist()}")
        X = X.join(wide_df, how="left")

    # --------------------------
    # STEP 5 — Cleanup
    # --------------------------
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method="ffill").fillna(method="bfill")

    print(f"[Success] Final feature matrix shape: {X.shape}")
    return X



# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------

def train_models(X, y_dict):
    models = {}
    for horizon, y in y_dict.items():
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                class_weight="balanced",
                random_state=42
            ))
        ])
        clf.fit(X, y)
        models[horizon] = clf
    return models


# ------------------------------------------------------------
# PREDICT + EXPLAIN
# ------------------------------------------------------------

def predict_and_explain(models, X, top_n=12):
    last = X.iloc[-1:]
    results = {}

    for horizon, model in models.items():
        probs = model.predict_proba(last)[0]
        classes = model.classes_
        pred = classes[np.argmax(probs)]

        # Extract feature importances
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
        feat_idx = np.argsort(importances)[::-1][:top_n]

        top_features = [
            {"feature": X.columns[i], "impact": float(importances[i])}
            for i in feat_idx
        ]

        results[horizon] = {
            "predicted_phase": pred,
            "probabilities": dict(zip(classes, probs)),
            "top_features": top_features
        }

    return results
