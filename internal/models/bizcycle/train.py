from db_loader import load_indicator_data, load_sp500_from_db
from preprocess import preprocess_monthly
from labelling import label_business_cycle
from model import create_future_targets, make_features
from composite_score import compute_composite_score
from enhance_model import (
    to_wide_monthly,
    make_features,
    train_xgb_multi_horizon,
    predict_and_explain,
)
import pandas as pd
from sqlalchemy import create_engine
from config import get_database_url
from dotenv import load_dotenv
import os

# ---------------------------------------------
# Load .env
# ---------------------------------------------
load_dotenv("C:/Users/harte/Documents/goTutorial/internal/.env")

# ---------------------------------------------
# 1. Load indicator data
# ---------------------------------------------
df = load_indicator_data()
print(df.head())

# ---------------------------------------------
# 2. Convert indicators → Monthly frequency
# ---------------------------------------------
monthly = preprocess_monthly(df)
print("\nMonthly Preprocessed")
print(monthly.head())

# ---------------------------------------------
# 3. Load SP500 from database
# ---------------------------------------------
engine = create_engine(get_database_url())
sp500 = load_sp500_from_db()
print("\nSP500 Raw:")
print(sp500.head())

# ---------------------------------------------
# 4. Label business cycle phases
# ---------------------------------------------
labeled_df = label_business_cycle(monthly, sp500)
labeled_df = labeled_df[~labeled_df.index.duplicated(keep="last")]
print("\nBusiness Cycle Labeled")

print(labeled_df.head())

# ---------------------------------------------
# 5. Create future prediction targets (1m, 3m, 6m)
# ---------------------------------------------
labeled_df = create_future_targets(labeled_df)
print("\nFuture Targets Created")
print(labeled_df.head())

# ---------------------------------------------
# 6. Compute Composite Scores for each horizon
# ---------------------------------------------
for horizon in ["cycle_1m", "cycle_3m", "cycle_6m"]:
    labeled_df, weights = compute_composite_score(labeled_df, horizon)
    print(f"\n=== Composite Indicator Weights for {horizon} ===")
    print(weights.sort_values(ascending=False).head(15))

print("\nComposite scores added to labeled_df!")
print(labeled_df.head())

# ---------------------------------------------
# 7. Convert into WIDE format for ML
# ---------------------------------------------
# 7. Convert long → wide so each indicator is a column
wide_df = labeled_df.copy()
wide_df = wide_df[~wide_df.index.duplicated(keep='first')]

# 8. Build model-ready features
X = make_features(wide_df)
X = X.loc[~X.index.duplicated()]
# 9. Build targets aligned with X
labeled_df = labeled_df.loc[~labeled_df.index.duplicated()]
df_targets = labeled_df[["cycle_1m","cycle_3m","cycle_6m"]].reindex(X.index)

# 10. Train multi-horizon models
pipelines, explainers = train_xgb_multi_horizon(X, df_targets)


# ---------------------------------------------
# 10. Make prediction on the latest data & explain it
# ---------------------------------------------
print("\n\n====================")
print("   MODEL PREDICTION ")
print("====================")

latest_prediction = predict_and_explain(pipelines, X, wide_df)

print("\n\nFinal Prediction Output:")
print(latest_prediction)
import json
output_path = r"C:\Users\harte\Documents\goTutorial\internal\model_debug_report.html"

with open("model_debug_report.html", "w") as f:
    f.write("<h1>Model Debug Report</h1>")
    f.write("<h2>Prediction Output</h2>")
    f.write("<pre>" + json.dumps(latest_prediction, indent=4) + "</pre>")
