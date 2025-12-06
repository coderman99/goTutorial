try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when xgboost is unavailable
    from sklearn.ensemble import GradientBoostingClassifier

    _XGBOOST_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder

TARGETS = ["cycle_1m", "cycle_3m", "cycle_6m"]


def _encode_categoricals(df):
    """Encode object/categorical columns with LabelEncoder."""

    encoders = {}
    df_encoded = df.copy()
    cat_cols = [
        c
        for c in df_encoded.columns
        if df_encoded[c].dtype == "object" or df_encoded[c].dtype.name == "category"
    ]

    for col in cat_cols:
        enc = LabelEncoder()
        enc.fit(df_encoded[col].astype(str).fillna("<NA>"))
        df_encoded[col] = enc.transform(df_encoded[col].astype(str).fillna("<NA>"))
        encoders[col] = enc

    return df_encoded, encoders

def make_features(wide_df):
    df = wide_df.copy()

    # forward/backward fill
    df = df.ffill().bfill()

    # Lag all indicators so model does not peek ahead
    for col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)

    # Drop NA created by lags
    df = df.dropna()

    # Extract label columns
    y_dict = {
        "cycle_1m": df["cycle_1m"].copy(),
        "cycle_3m": df["cycle_3m"].copy(),
        "cycle_6m": df["cycle_6m"].copy()
    }

    # Drop label columns from features
    X = df.drop(columns=["cycle_1m", "cycle_3m", "cycle_6m"])

    return X, y_dict


def prepare_features(df):
    """
    Remove non-numeric columns and prepare X, y.
    """

    # ---- DROP STRING COLUMNS ----
    drop_cols = ["name", "series_id", "indicator_cat", "cycle_phase"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # ---- Encode targets (Expansion / Contraction / Peak) ----
    encoders = {}
    for t in TARGETS:
        enc = LabelEncoder()
        df[t] = enc.fit_transform(df[t])
        encoders[t] = enc

    # ---- Select feature columns (everything except targets) ----
    X_raw = df.drop(columns=TARGETS)
    y = df[TARGETS].copy()

    # Encode any categorical feature columns
    X_encoded, _ = _encode_categoricals(X_raw)

    # Fill missing numeric values
    X = X_encoded.ffill().bfill()

    return X, y, encoders

def train_model(df):
    y = df["cycle_phase"]
    X = df.drop(columns=["cycle_phase"])

    X, _ = _encode_categoricals(X)

    X = X.fillna(method="ffill").fillna(method="bfill")

    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, shuffle=False
    )

    if _LIGHTGBM_AVAILABLE:
        class_counts = np.bincount(y_encoded)
        class_weight = {
            cls: class_counts.sum() / (len(class_counts) * cnt)
            for cls, cnt in enumerate(class_counts) if cnt > 0
        }

        model = LGBMClassifier(
            n_estimators=400,
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            class_weight=class_weight,
            random_state=42,
            importance_type="gain",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

    preds_encoded = model.predict(X_test)
    preds = label_encoder.inverse_transform(preds_encoded)
    y_test_labels = label_encoder.inverse_transform(y_test)

    print(classification_report(y_test_labels, preds))
    return model, X_test, y_test_labels

def create_future_targets(df):
    df["cycle_1m"] = df["cycle_phase"].shift(-1)
    df["cycle_3m"] = df["cycle_phase"].shift(-3)
    df["cycle_6m"] = df["cycle_phase"].shift(-6)
    return df

def train_multi_horizon(df):
    """
    Trains 3 separate models for 1m / 3m / 6m targets.
    """

    X, y, encoders = prepare_features(df)

    models = {}

    for target in TARGETS:

        print(f"\n=== Training model for {target} ===")

        y_target = y[target]

        # Ensure no missing labels
        valid = y_target.dropna().index
        X_valid = X.loc[valid]
        y_valid = y_target.loc[valid]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, shuffle=False
        )
        if _LIGHTGBM_AVAILABLE:
            class_counts = y_train.value_counts()
            class_weight = {
                cls: class_counts.sum() / (len(class_counts) * cnt)
                for cls, cnt in class_counts.items()
            }

            model = LGBMClassifier(
                n_estimators=400,
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                class_weight=class_weight,
                random_state=42,
                importance_type="gain",
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="multi_logloss",
                verbose=False,
            )
        else:
            model = GradientBoostingClassifier(random_state=42)

            model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print(f"{target} accuracy: {acc:.3f}")

        models[target] = model

    return models, encoders
