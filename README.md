
# Nifty Market Terminal

Bloomberg-inspired, retail-friendly technical screening terminal for the
Nifty Total Market universe.

## Architecture

The app uses one shared market scan:

1. Download the current universe.
2. Batch-download daily OHLCV data.
3. Calculate all technical indicators once.
4. Reuse the cached result across strategy pages.
5. Fetch fundamentals only for final candidates.

This keeps the app much lighter than running a separate data download for each
strategy.

## Current strategy modules

- Market Regime: 50/200 SMA context
- 9/21 Momentum: short-term momentum
- 20/50 Swing: medium-term structure
- EMA 255 Pullback: RSI < 35 and within ±2% of EMA 255
- Convergence: Trend Score + Entry Score
- Final Buying List: top research candidates with fundamentals

## Data sources

- NSE index constituent files for the universe
- Yahoo Finance daily OHLCV via `yfinance`

No broker API key is required.

## Deploy

Upload these files to a GitHub repository:

```text
.
├── streamlit_app.py
├── engine.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

Then deploy `streamlit_app.py` on Streamlit Community Cloud.

## Notes

The scanner is decision-support software. It is not a trading system or
investment recommendation. Crossover and RSI signals should be validated with
out-of-sample testing before real-money use.
