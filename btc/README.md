# BTC data pipeline

This folder contains a small Python pipeline that downloads Bitcoin market data and fear/greed sentiment, calculates technical indicators, and stores the results in a SQLite database.

## Requirements

Install the Python dependencies (preferrably in a virtual environment):

```
pip install -r requirements.txt
```

## Running

Execute the pipeline from the repository root:

```
python -m btc.main
```

The script will download approximately seven years of data, compute:

- 20-day exponential moving average (EMA)
- 20-day Bollinger Bands (middle/upper/lower)
- 14-day Stochastic RSI

Results are stored in `btc/btc_data.sqlite` in the `btc_metrics` table. Existing rows are updated based on date.
