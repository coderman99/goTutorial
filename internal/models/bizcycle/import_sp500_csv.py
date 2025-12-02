import pandas as pd
from sqlalchemy import create_engine
from config import get_database_url

CSV_PATH = "C:/Users/harte/Documents/goTutorial/internal/models/bizcycle/sp500_monthly.csv"

def load_sp500_csv_to_db():
    df = pd.read_csv(CSV_PATH)

    # Convert YYYY-MM to end-of-month timestamps
    df["timestamp"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    # Rename columns to match DB schema
    df["name"] = "StockMarketIndex"
    df["series_id"] = "SP500"
    df["value"] = df["SPX_Close"]
    df["indicator_cat"] = "Leading"

    # Select correct DB columns
    df = df[["name", "series_id", "value", "timestamp", "indicator_cat"]]

    engine = create_engine(get_database_url())
    df.to_sql("indicator_models", engine, if_exists="append", index=False)

    print("Inserted", len(df), "SP500 monthly rows into indicator_models")

if __name__ == "__main__":
    load_sp500_csv_to_db()
