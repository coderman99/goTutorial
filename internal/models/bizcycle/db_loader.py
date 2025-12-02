# db_loader.py
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)


def load_indicator_data():
    """
    Loads all indicators EXCEPT SP500.
    """
    df = pd.read_sql("SELECT * FROM indicator_models", engine)

    # Remove SP500 because we will load it separately
    df = df[df["series_id"] != "SP500"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    return df


def load_sp500_from_db():
    """
    Loads SP500 from the same table.
    Converts daily → monthly frequency like your script expected.
    """
    spx = pd.read_sql("""
        SELECT *
        FROM indicator_models
        WHERE series_id = 'SP500'
    """, engine)

    if spx.empty:
        raise ValueError("ERROR: SP500 not found in DB. Check series_id or name.")

    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)
    spx = spx.sort_values("timestamp")

    # Convert daily → month-end (your previous code)
    spx = spx.set_index("timestamp").resample("M").last()

    spx = spx.rename(columns={"value": "sp500"})
    return spx[["sp500"]]
