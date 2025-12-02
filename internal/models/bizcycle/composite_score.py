import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def compute_indicator_weights(X, y):
    """Compute predictive weights for each indicator."""

    # ANOVA F-scores
    f_scores, _ = f_classif(X, y)

    # Mutual information
    mi_scores = mutual_info_classif(X, y)

    # Random forest importance
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)
    rf_scores = rf.feature_importances_

    # Normalize everything to weight scale
    f_norm = f_scores / f_scores.sum()
    mi_norm = mi_scores / mi_scores.sum()
    rf_norm = rf_scores / rf_scores.sum()

    # Composite weight
    weights = (f_norm + mi_norm + rf_norm) / 3

    return pd.Series(weights, index=X.columns)


def compute_composite_score(df, horizon):

    # 1. Target series
    y = df[horizon]

    # 2. Keep ONLY numeric columns for scoring
    X = df.drop(columns=[horizon])
    X = X.select_dtypes(include=["number"])     # <— FIX

    # --------------------------------------------
    # Remove rows where target is NA
    # --------------------------------------------
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # --------------------------------------------
    # Fill NA (ANOVA cannot handle NaN)
    # --------------------------------------------
    X = X.ffill().bfill()
    X = X.dropna()

    # Align y to X after cleaning
    y = y.loc[X.index]

    # --------------------------------------------
    # Scale numeric features
    # --------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------
    # Compute ANOVA F-scores
    # --------------------------------------------
    f_scores, _ = f_classif(X_scaled, y)

    weights = pd.Series(f_scores, index=X.columns)
    weights = weights / weights.sum()

    # --------------------------------------------
    # Add composite score to main dataframe
    # --------------------------------------------
    df.loc[X.index, f"composite_{horizon}"] = (X * weights).sum(axis=1)

    return df, weights