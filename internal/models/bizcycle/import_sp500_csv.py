import pandas as pd
from sqlalchemy import create_engine
from config import get_database_url

CSV_PATH = "C:/Users/harte/Documents/goTutorial/internal/models/bizcycle/sp500_monthly.csv"

def load_sp500_csv_to_db():
    df = pd.read_csv(CSV_PATH)

    # Convert YYYY-MM to end-of-month timestamps
    df["timestamp"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    # Rename columns to match sp500_models schema
    df = df.rename(columns={"SPX_Close": "close"})
    df["id"] = range(1, len(df) + 1)

    # Select correct DB columns
    df = df[["id", "timestamp", "close"]]

    engine = create_engine(get_database_url())
    df.to_sql("sp500_models", engine, if_exists="append", index=False)

    print("Inserted", len(df), "SP500 monthly rows into sp500_models")

if __name__ == "__main__":
    load_sp500_csv_to_db()
