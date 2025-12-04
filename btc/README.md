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

## Training the trading model

You can train a lightweight model that issues a buy/hold/sell suggestion and estimates how long to hold the position. The labels are derived from forward returns and holding duration is measured until a ±4% move or a 45-day cap.

```bash
python - <<'PY'
from btc.model import train_and_predict

bundle, signal = train_and_predict()
print("Action reports:\n", bundle.reports["classification"])
print("Hold MAE:", bundle.reports["hold_mae"])
print("Latest signal:", signal)
PY
```

The helper will fetch seven years of data, train the models, and print both validation metrics and the most recent signal (action plus expected holding days).
