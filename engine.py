
from __future__ import annotations

import io
import time
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import requests
import yfinance as yf
import streamlit as st


NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/151.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}

# NSE's current Total Market page defines the universe as Nifty 500 +
# Nifty Microcap 250. The app still validates the final count dynamically.
TOTAL_MARKET_URLS = []

NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]

MICROCAP250_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
]


def _read_nse_csv(urls: list[str]) -> pd.DataFrame:
    errors = []

    for url in urls:
        try:
            r = requests.get(url, headers=NSE_HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))

            normalized = {
                str(c).strip().lower(): c
                for c in df.columns
            }

            symbol_col = normalized.get("symbol")
            company_col = (
                normalized.get("company name")
                or normalized.get("company")
                or normalized.get("name")
            )

            if not symbol_col:
                raise ValueError(
                    f"Symbol column not found. Columns: {df.columns.tolist()}"
                )

            cols = [symbol_col]
            if company_col:
                cols.append(company_col)

            result = df[cols].copy()
            result = result.rename(columns={symbol_col: "Symbol"})

            if company_col:
                result = result.rename(columns={company_col: "Company"})
            else:
                result["Company"] = result["Symbol"]

            result["Symbol"] = (
                result["Symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
            )
            result["Company"] = (
                result["Company"]
                .astype(str)
                .str.strip()
            )

            result = (
                result
                .dropna(subset=["Symbol"])
                .drop_duplicates("Symbol")
                .reset_index(drop=True)
            )

            if len(result) < 100:
                raise ValueError(
                    f"Only {len(result)} constituents returned."
                )

            result["Yahoo Symbol"] = result["Symbol"] + ".NS"
            return result

        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError("NSE universe download failed: " + " | ".join(errors))


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_universe(universe_name: str = "NIFTY TOTAL MARKET") -> pd.DataFrame:
    if universe_name == "NIFTY 500":
        return _read_nse_csv(NIFTY500_URLS)

    # The official Total Market universe is Nifty 500 plus
    # Nifty Microcap 250. Build it from the two public constituent files.
    # This avoids relying on a non-public/unstable direct Total Market CSV URL.
    nifty500 = _read_nse_csv(NIFTY500_URLS)
    microcap = _read_nse_csv(MICROCAP250_URLS)

    result = (
        pd.concat([nifty500, microcap], ignore_index=True)
        .drop_duplicates("Symbol")
        .sort_values("Symbol")
        .reset_index(drop=True)
    )

    return result


def _normalize_yfinance(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                continue

            frame = data[ticker].copy().dropna(how="all")
            if frame.empty:
                continue

            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(frame.columns):
                continue

            frame = frame.reset_index()
            frame["Yahoo Symbol"] = ticker
            frames.append(frame)
    else:
        if len(tickers) == 1:
            frame = data.copy().dropna(how="all")
            if not frame.empty:
                frame = frame.reset_index()
                frame["Yahoo Symbol"] = tickers[0]
                frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    out["Date"] = pd.to_datetime(
        out["Date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return (
        out
        .dropna(subset=["Date", "High", "Low", "Close"])
        .drop_duplicates(["Date", "Yahoo Symbol"])
        .sort_values(["Yahoo Symbol", "Date"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=4 * 60 * 60, show_spinner=False)
def download_price_batch(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    symbols = list(tickers)

    for attempt in range(3):
        try:
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )

            normalized = _normalize_yfinance(data, symbols)
            if not normalized.empty:
                return normalized
        except Exception:
            pass

        time.sleep(1.0 + attempt * 1.5)

    return pd.DataFrame()


def download_prices(
    universe: pd.DataFrame,
    years: int = 4,
    batch_size: int = 75,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25))

    tickers = universe["Yahoo Symbol"].tolist()
    all_frames: list[pd.DataFrame] = []
    failures: list[str] = []

    total = (len(tickers) + batch_size - 1) // batch_size

    for batch_no, idx in enumerate(
        range(0, len(tickers), batch_size),
        start=1,
    ):
        batch = tickers[idx: idx + batch_size]

        frame = download_price_batch(
            tuple(batch),
            start.isoformat(),
            end.isoformat(),
        )

        returned = (
            set(frame["Yahoo Symbol"].unique())
            if not frame.empty
            else set()
        )

        if not frame.empty:
            all_frames.append(frame)

        missing = [t for t in batch if t not in returned]

        # One fallback attempt for missing tickers. We do not loop indefinitely.
        for ticker in missing:
            one = download_price_batch(
                (ticker,),
                start.isoformat(),
                end.isoformat(),
            )

            if one.empty:
                failures.append(ticker)
            else:
                all_frames.append(one)

        if progress_callback:
            progress_callback(batch_no, total, len(failures))

        # Small pause helps avoid hammering the free public endpoint.
        time.sleep(0.15)

    if not all_frames:
        return pd.DataFrame(), sorted(set(failures))

    prices = (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates(["Date", "Yahoo Symbol"], keep="last")
        .sort_values(["Yahoo Symbol", "Date"])
        .reset_index(drop=True)
    )

    return prices, sorted(set(failures))


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


# -------------------------------------------------------------------
# Feature configuration
# -------------------------------------------------------------------
RS_PERIODS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
LIQUIDITY_THRESHOLD = 1_00_00_000  # ₹1 crore average daily traded value


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, pd.NA)


def _liquidity_bucket(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    if value < 1_00_00_000:
        return "Illiquid"
    if value < 5_00_00_000:
        return "Low Liquidity"
    if value < 25_00_00_000:
        return "Tradeable"
    if value < 100_00_00_000:
        return "Liquid"
    return "Highly Liquid"


def calculate_indicators(
    prices: pd.DataFrame,
    ema_long: int = 255,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Calculate the shared technical, liquidity and quality feature set once."""
    if prices.empty:
        return pd.DataFrame()

    required = {"Date", "Yahoo Symbol", "High", "Low", "Close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    result_frames = []

    for ticker, frame in prices.groupby("Yahoo Symbol", sort=False):
        frame = frame.sort_values("Date").drop_duplicates("Date").copy()
        close = pd.to_numeric(frame["Close"], errors="coerce")
        high = pd.to_numeric(frame["High"], errors="coerce")
        low = pd.to_numeric(frame["Low"], errors="coerce")
        volume = pd.to_numeric(frame.get("Volume", pd.Series(index=frame.index, dtype=float)), errors="coerce")

        frame["EMA9"] = close.ewm(span=9, adjust=False).mean()
        frame["EMA21"] = close.ewm(span=21, adjust=False).mean()
        frame["SMA20"] = close.rolling(20, min_periods=20).mean()
        frame["SMA50"] = close.rolling(50, min_periods=50).mean()
        frame["SMA200"] = close.rolling(200, min_periods=200).mean()
        frame[f"EMA{ema_long}"] = close.ewm(
            span=ema_long, adjust=False, min_periods=ema_long
        ).mean()
        frame[f"RSI{rsi_period}"] = rsi_wilder(close, rsi_period)

        # ATR 14. True Range uses the previous close, so no future data is used.
        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        frame["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        frame["ATRPercent"] = _safe_ratio(frame["ATR14"], close) * 100

        # Volume and liquidity context.
        frame["VolumeSMA20"] = volume.rolling(20, min_periods=20).mean()
        frame["VolumeRatio"] = _safe_ratio(volume, frame["VolumeSMA20"])
        frame["TradedValue"] = close * volume
        frame["AvgTradedValue20"] = frame["TradedValue"].rolling(20, min_periods=20).mean()

        # Cross states. Yesterday vs today only.
        frame["Cross9_21"] = (frame["EMA9"] > frame["EMA21"]) & (
            frame["EMA9"].shift(1) <= frame["EMA21"].shift(1)
        )
        frame["Cross20_50"] = (frame["SMA20"] > frame["SMA50"]) & (
            frame["SMA20"].shift(1) <= frame["SMA50"].shift(1)
        )
        frame["Cross50_200"] = (frame["SMA50"] > frame["SMA200"]) & (
            frame["SMA50"].shift(1) <= frame["SMA200"].shift(1)
        )

        frame["BullMomentum"] = frame["EMA9"] > frame["EMA21"]
        frame["BullSwing"] = frame["SMA20"] > frame["SMA50"]
        frame["BullRegime"] = frame["SMA50"] > frame["SMA200"]

        frame["EMA255DistancePct"] = _safe_ratio(
            close - frame[f"EMA{ema_long}"], frame[f"EMA{ema_long}"]
        ) * 100
        frame["Pullback"] = (
            (frame[f"RSI{rsi_period}"] < 35)
            & (frame["EMA255DistancePct"].abs() <= 2)
        )

        frame["MomentumFresh"] = frame["Cross9_21"]
        frame["SwingFresh"] = frame["Cross20_50"]
        frame["RegimeFresh"] = frame["Cross50_200"]

        # Relative performance. Percentile ranking is applied across the scanned universe later.
        for label, periods in RS_PERIODS.items():
            frame[f"Return{label}"] = close.pct_change(periods=periods) * 100

        frame["DailyReturnPct"] = close.pct_change() * 100
        frame["GapPct"] = _safe_ratio(frame.get("Open", close) - prev_close, prev_close) * 100
        frame["VolumeConfirmedMomentum"] = (
            frame["BullMomentum"]
            & (frame["DailyReturnPct"] > 0)
            & (frame["VolumeRatio"] >= 1.5)
        )

        previous_20_high = high.rolling(20, min_periods=20).max().shift(1)
        frame["Breakout20"] = (
            (close > previous_20_high)
            & (frame["VolumeRatio"] >= 1.5)
            & frame["BullSwing"]
        )

        frame["Yahoo Symbol"] = ticker
        result_frames.append(frame)

    return pd.concat(result_frames, ignore_index=True)


def latest_snapshot(
    indicators: pd.DataFrame,
    universe: pd.DataFrame,
    ema_long: int = 255,
    rsi_period: int = 14,
) -> pd.DataFrame:
    if indicators.empty:
        return pd.DataFrame()

    meta = universe.set_index("Yahoo Symbol")[["Symbol", "Company"]].to_dict("index")
    rows = []
    latest_columns = [
        "Close", "EMA9", "EMA21", "SMA20", "SMA50", "SMA200",
        f"EMA{ema_long}", f"RSI{rsi_period}", "EMA255DistancePct",
        "BullMomentum", "BullSwing", "BullRegime", "MomentumFresh",
        "SwingFresh", "RegimeFresh", "Pullback", "ATR14", "ATRPercent",
        "VolumeSMA20", "VolumeRatio", "AvgTradedValue20", "DailyReturnPct",
        "GapPct", "VolumeConfirmedMomentum", "Breakout20",
    ] + [f"Return{label}" for label in RS_PERIODS]

    for ticker, frame in indicators.groupby("Yahoo Symbol", sort=False):
        row = frame.sort_values("Date").iloc[-1]
        company = meta.get(ticker, {"Symbol": ticker.replace(".NS", ""), "Company": ""})
        item = {"Symbol": company["Symbol"], "Company": company["Company"], "Yahoo Symbol": ticker, "Date": row["Date"]}
        for col in latest_columns:
            item[col] = row.get(col, pd.NA)
        rows.append(item)

    snapshot = pd.DataFrame(rows)
    bool_cols = [
        "BullMomentum", "BullSwing", "BullRegime", "MomentumFresh", "SwingFresh",
        "RegimeFresh", "Pullback", "VolumeConfirmedMomentum", "Breakout20",
    ]
    for col in bool_cols:
        snapshot[col] = snapshot[col].fillna(False).astype(bool)

    # Cross-sectional relative strength percentiles. 100 means strongest in the scan.
    for label in RS_PERIODS:
        ret_col = f"Return{label}"
        snapshot[f"RS{label}Pct"] = snapshot[ret_col].rank(pct=True, method="average") * 100

    snapshot["LiquidityBucket"] = snapshot["AvgTradedValue20"].apply(_liquidity_bucket)
    snapshot["LiquidityEligible"] = snapshot["AvgTradedValue20"] >= LIQUIDITY_THRESHOLD
    return snapshot


def add_days_since_cross(
    indicators: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Days since the latest bullish crossover for each stock."""
    if indicators.empty or snapshot.empty:
        return snapshot

    cross_dates = {}
    for ticker, frame in indicators.groupby("Yahoo Symbol", sort=False):
        latest_date = frame["Date"].max()
        cross_dates[ticker] = {}
        for cross_col, out_col in [
            ("Cross9_21", "DaysSince9_21"),
            ("Cross20_50", "DaysSince20_50"),
            ("Cross50_200", "DaysSince50_200"),
        ]:
            dates = frame.loc[frame[cross_col].fillna(False), "Date"]
            cross_dates[ticker][out_col] = int((latest_date - dates.iloc[-1]).days) if not dates.empty else None

    extra = pd.DataFrame.from_dict(cross_dates, orient="index")
    extra.index.name = "Yahoo Symbol"
    return snapshot.join(extra, on="Yahoo Symbol")


def convergence_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Score distinct setup paths instead of averaging contradictory signals."""
    if snapshot.empty:
        return pd.DataFrame()

    df = snapshot.copy()
    rs3 = df["RS3MPct"].fillna(0)
    volume = df["VolumeRatio"].fillna(0)
    liquid = df["LiquidityEligible"].fillna(False).astype(int)

    # Baseline trend evidence retained for continuity with the existing app.
    df["RegimeScore"] = df["BullRegime"].astype(int) * 25
    df["MomentumScore"] = df["BullMomentum"].astype(int) * 20 + df["MomentumFresh"].astype(int) * 5
    df["SwingScore"] = df["BullSwing"].astype(int) * 20 + df["SwingFresh"].astype(int) * 5
    near_ema = df["EMA255DistancePct"].abs() <= 2
    rsi = df.filter(regex=r"^RSI\d+$").iloc[:, 0]
    df["EntryScore"] = near_ema.astype(int) * 15 + ((rsi >= 35) & (rsi <= 60)).astype(int) * 10
    df["TrendScore"] = df["RegimeScore"] + df["MomentumScore"] + df["SwingScore"]

    # Setup-specific paths. Each path is evaluated on its own logic.
    df["TrendContinuationScore"] = (
        df["BullRegime"].astype(int) * 30
        + df["BullSwing"].astype(int) * 25
        + df["BullMomentum"].astype(int) * 20
        + (rs3 >= 70).astype(int) * 15
        + liquid * 10
    )
    df["PullbackScore"] = (
        df["BullRegime"].astype(int) * 25
        + df["BullSwing"].astype(int) * 15
        + df["Pullback"].astype(int) * 30
        + (rs3 >= 50).astype(int) * 10
        + liquid * 10
        + (volume >= 0.8).astype(int) * 10
    )
    df["FreshMomentumScore"] = (
        df["BullRegime"].astype(int) * 20
        + df["BullSwing"].astype(int) * 15
        + df["BullMomentum"].astype(int) * 15
        + df["MomentumFresh"].astype(int) * 15
        + df["VolumeConfirmedMomentum"].astype(int) * 15
        + (rs3 >= 70).astype(int) * 10
        + liquid * 10
    )
    df["BreakoutScore"] = (
        df["BullRegime"].astype(int) * 20
        + df["BullSwing"].astype(int) * 20
        + df["Breakout20"].astype(int) * 25
        + (volume >= 1.5).astype(int) * 10
        + (rs3 >= 70).astype(int) * 15
        + liquid * 10
    )

    score_cols = ["TrendContinuationScore", "PullbackScore", "FreshMomentumScore", "BreakoutScore"]
    setup_labels = {
        "TrendContinuationScore": "Trend Continuation",
        "PullbackScore": "Pullback in Bull Regime",
        "FreshMomentumScore": "Fresh Momentum",
        "BreakoutScore": "Volume Breakout",
    }
    df["SetupScore"] = df[score_cols].max(axis=1)
    best_col = df[score_cols].idxmax(axis=1)
    df["Setup"] = best_col.map(setup_labels)

    # Only label an active setup when its defining trigger exists.
    active = (
        (df["Setup"] == "Pullback in Bull Regime") & df["Pullback"]
    ) | (
        (df["Setup"] == "Fresh Momentum") & df["MomentumFresh"]
    ) | (
        (df["Setup"] == "Volume Breakout") & df["Breakout20"]
    ) | (
        (df["Setup"] == "Trend Continuation") & df["BullRegime"] & df["BullSwing"] & df["BullMomentum"]
    )
    df.loc[~active, "Setup"] = "No active setup"

    # ConvergenceScore remains the public ranking field, now setup-aware.
    df["ConvergenceScore"] = df["SetupScore"].round(0).astype(int)
    return df.sort_values(
        ["ConvergenceScore", "RS3MPct", "AvgTradedValue20"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

def fundamental_snapshot(ticker: str) -> dict:
    """Fetch six late-stage fundamental context fields only for finalists."""
    defaults = {
        "PE": None,
        "Revenue Growth %": None,
        "Profit Margin %": None,
        "Debt/Equity": None,
        "EV/EBITDA": None,
        "Market Cap": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
        profit = info.get("profitMargins")
        revenue_growth = info.get("revenueGrowth")
        return {
            "PE": info.get("trailingPE"),
            "Revenue Growth %": revenue_growth * 100 if revenue_growth is not None else None,
            "Profit Margin %": profit * 100 if profit is not None else None,
            "Debt/Equity": info.get("debtToEquity"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "Market Cap": info.get("marketCap"),
        }
    except Exception:
        return defaults
