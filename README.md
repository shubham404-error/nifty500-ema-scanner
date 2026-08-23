# Nifty 500 EMA Scanner

A Streamlit Community Cloud-ready application that scans the current Nifty 500
for stocks that most recently touched a selected exponential moving average.

## Features

- No API key
- Current Nifty 500 constituent download with fallback URL
- Automatic NSE to Yahoo ticker mapping: `SYMBOL.NS`
- Batch historical downloads
- Retry logic for failed batches and individual tickers
- Cached constituent and market-data requests
- Configurable EMA period
- Wick-touch or close-near-EMA signal definitions
- Latest occurrence retained for every stock
- Ranking by newest occurrence date
- Interactive filtering
- CSV download
- Failed ticker report

## Repository layout

```text
.
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the entire project, keeping `app.py` and `requirements.txt` in the root.
3. Open Streamlit Community Cloud.
4. Create an app and select the GitHub repository.
5. Select the `main` branch and `app.py`.
6. Deploy.

No secrets or API keys are required.

## Signal definition

### Wick touch

```text
Low <= EMA <= High
```

### Close near EMA

```text
abs(Close - EMA) / EMA <= tolerance
```

Only the most recent qualifying touch for each stock is retained.

## Notes

The app uses `yfinance` to access public Yahoo Finance data. Availability,
ticker coverage, and response reliability can occasionally vary. The app reports
tickers that remain unresolved after retries instead of failing the entire scan.
