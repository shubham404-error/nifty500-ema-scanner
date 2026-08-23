
from __future__ import annotations

import io
import time
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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

NIFTY50_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
]

NIFTY200_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
]

NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]

MICROCAP250_URLS = [
    # Primary official URL. The archive URL used in older builds returned 404.
    "https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
    "https://niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv",
]


def _read_nse_csv(urls: list[str], min_constituents: int = 100) -> pd.DataFrame:
    """Read an official NSE/Nifty constituent CSV with bounded retries.

    A source is accepted only after schema and minimum-size validation. This
    prevents an HTML error page or partial response from silently becoming the
    market universe.
    """
    errors: list[str] = []
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    for url in urls:
        for attempt in range(3):
            try:
                response = session.get(url, timeout=(10, 30))
                response.raise_for_status()
                text = response.content.decode("utf-8-sig", errors="replace")
                if "<html" in text[:500].lower():
                    raise ValueError("received HTML instead of CSV")

                df = pd.read_csv(io.StringIO(text))
                normalized = {str(c).strip().lower(): c for c in df.columns}
                symbol_col = normalized.get("symbol")
                company_col = (
                    normalized.get("company name")
                    or normalized.get("company")
                    or normalized.get("name")
                )
                if not symbol_col:
                    raise ValueError(f"Symbol column not found. Columns: {df.columns.tolist()}")

                cols = [symbol_col] + ([company_col] if company_col else [])
                result = df[cols].copy().rename(columns={symbol_col: "Symbol"})
                result["Company"] = result[company_col] if company_col else result["Symbol"]
                if company_col and company_col in result.columns:
                    result = result.drop(columns=[company_col])

                result["Symbol"] = result["Symbol"].astype(str).str.strip().str.upper()
                result["Company"] = result["Company"].astype(str).str.strip()
                result = result.loc[result["Symbol"].notna() & result["Symbol"].ne("")]
                result = result.drop_duplicates("Symbol").reset_index(drop=True)
                if len(result) < min_constituents:
                    raise ValueError(
                        f"Only {len(result)} valid constituents returned; expected at least {min_constituents}."
                    )

                result["Yahoo Symbol"] = result["Symbol"] + ".NS"
                return result[["Symbol", "Company", "Yahoo Symbol"]]
            except Exception as exc:
                if attempt == 2:
                    errors.append(f"{url}: {exc}")
                else:
                    time.sleep((0.75 * (2 ** attempt)) + random.uniform(0.0, 0.25))

    raise RuntimeError("NSE universe download failed: " + " | ".join(errors))


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_universe(universe_name: str = "NIFTY TOTAL MARKET") -> pd.DataFrame:
    """Load an official Nifty constituent universe.

    Universe selection is independent from strategy/scoring logic.
    """
    if universe_name == "NIFTY 50":
        return _read_nse_csv(NIFTY50_URLS, min_constituents=45)
    if universe_name == "NIFTY 200":
        return _read_nse_csv(NIFTY200_URLS, min_constituents=180)
    if universe_name == "NIFTY 500":
        return _read_nse_csv(NIFTY500_URLS, min_constituents=450)

    nifty500 = _read_nse_csv(NIFTY500_URLS, min_constituents=450)
    microcap = _read_nse_csv(MICROCAP250_URLS, min_constituents=200)
    result = (
        pd.concat([nifty500, microcap], ignore_index=True)
        .drop_duplicates("Symbol")
        .sort_values("Symbol")
        .reset_index(drop=True)
    )
    if len(result) < 650:
        raise RuntimeError(f"Total Market universe appears incomplete: only {len(result)} unique stocks.")
    return result

def _normalize_yfinance(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output while preserving adjusted and raw prices.

    Technical indicators use adjusted OHLC. Liquidity uses RawClose * Volume so
    historical split adjustments do not distort traded-value calculations.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    def _extract_ticker_frame(source: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if not isinstance(source.columns, pd.MultiIndex):
            return source.copy() if len(tickers) == 1 else pd.DataFrame()

        level0 = source.columns.get_level_values(0)
        level1 = source.columns.get_level_values(1)
        if ticker in level0:
            return source[ticker].copy()
        if ticker in level1:
            return source.xs(ticker, axis=1, level=1, drop_level=True).copy()
        return pd.DataFrame()

    for ticker in tickers:
        frame = _extract_ticker_frame(data, ticker).dropna(how="all")
        if frame.empty:
            continue

        # auto_adjust=False gives both raw Close and Adj Close. Build an
        # adjusted OHLC series explicitly and retain RawClose for liquidity.
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(frame.columns):
            continue

        frame["RawClose"] = pd.to_numeric(frame["Close"], errors="coerce")
        if "Adj Close" in frame.columns:
            adj = pd.to_numeric(frame["Adj Close"], errors="coerce")
            raw = frame["RawClose"].replace(0, pd.NA)
            adjustment = (adj / raw).replace([float("inf"), float("-inf")], pd.NA)
            # Only adjust rows where Yahoo supplied a valid adjustment factor.
            for col in ["Open", "High", "Low", "Close"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce") * adjustment.fillna(1.0)
        else:
            for col in ["Open", "High", "Low", "Close"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame = frame.reset_index()
        date_col = "Date" if "Date" in frame.columns else frame.columns[0]
        if date_col != "Date":
            frame = frame.rename(columns={date_col: "Date"})
        frame["Yahoo Symbol"] = ticker
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

    for col in ["Open", "High", "Low", "Close", "RawClose", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Reject impossible or incomplete rows instead of allowing bad data into
    # every downstream indicator.
    valid = (
        out[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1))
        & (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))
    )

    return (
        out.loc[valid]
        .dropna(subset=["Date", "High", "Low", "Close"])
        .drop_duplicates(["Date", "Yahoo Symbol"], keep="last")
        .sort_values(["Yahoo Symbol", "Date"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def download_price_batch(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download one batch with bounded retries and normalized output."""
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return pd.DataFrame()

    for attempt in range(3):
        try:
            # Keep auto_adjust=False so RawClose and Adj Close are both
            # available. _normalize_yfinance creates adjusted OHLC explicitly.
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
            # Batch failure is handled by the caller's bounded fallback path.
            pass

        time.sleep((1.0 * (2 ** attempt)) + random.uniform(0.0, 0.5))

    return pd.DataFrame()

def download_prices(
    universe: pd.DataFrame,
    years: int = 4,
    batch_size: int = 75,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    # The scanner is intentionally end-of-day, not intraday. The current IST
    # session is excluded because Yahoo's daily candle can change while the
    # market is open or while providers finalize the session.
    #
    # yfinance's end date is exclusive, so using today's IST date keeps the
    # latest completed trading session as the common scan endpoint.
    end = datetime.now(ZoneInfo("Asia/Kolkata")).date()
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

    # A scan must use one common completed market date. Otherwise a partially
    # returned symbol can be ranked using yesterday while other symbols use the
    # latest session, causing unstable RSI/RS percentiles and setup changes.
    as_of_date = prices["Date"].max()
    latest_by_symbol = prices.groupby("Yahoo Symbol")["Date"].max()
    stale_symbols = latest_by_symbol[latest_by_symbol != as_of_date].index.tolist()
    if stale_symbols:
        failures.extend(stale_symbols)
        prices = prices.loc[
            ~prices["Yahoo Symbol"].isin(stale_symbols)
        ].copy()

    # A symbol with only a handful of rows is not a valid substitute for a
    # complete download. Keep it in failures so the scan can report coverage.
    row_counts = prices.groupby("Yahoo Symbol")["Date"].nunique()
    min_required = min(260, max(60, int(years * 252 * 0.5)))
    short_history = row_counts[row_counts < min_required].index.tolist()
    failures.extend(short_history)
    if short_history:
        prices = prices.loc[
            ~prices["Yahoo Symbol"].isin(short_history)
        ].copy()

    failures = sorted(set(failures))

    # Do not silently publish a materially incomplete universe. A partial
    # universe changes cross-sectional relative-strength percentiles and can
    # therefore change the candidate list even when market prices did not.
    expected = len(tickers)
    returned = prices["Yahoo Symbol"].nunique()
    coverage = returned / max(expected, 1)
    if coverage < 0.90:
        raise RuntimeError(
            f"Market-data coverage too low: {returned}/{expected} "
            f"({coverage:.1%}). Scan was discarded rather than producing "
            "a potentially unstable ranking. Please rerun later."
        )

    return prices, failures


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
        raw_close = pd.to_numeric(frame.get("RawClose", close), errors="coerce")
        frame["TradedValue"] = raw_close * volume
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
        ordered = frame.sort_values("Date")
        row = ordered.iloc[-1]
        company = meta.get(ticker, {"Symbol": ticker.replace(".NS", ""), "Company": ""})
        bars = len(ordered)
        item = {
            "Symbol": company["Symbol"],
            "Company": company["Company"],
            "Yahoo Symbol": ticker,
            "Date": row["Date"],
            "Bars": bars,
            "HistoryEligible": bars >= max(ema_long, max(RS_PERIODS.values())) + 1,
        }
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

    # Cross-sectional relative strength percentiles. Rank only observations
    # with sufficient history for that horizon, so missing data is not treated
    # as weak performance.
    for label, periods in RS_PERIODS.items():
        ret_col = f"Return{label}"
        eligible = snapshot["Bars"] >= periods + 1
        snapshot[f"RS{label}Pct"] = pd.NA
        snapshot.loc[eligible, f"RS{label}Pct"] = (
            snapshot.loc[eligible, ret_col].rank(pct=True, method="average") * 100
        )

    snapshot["LiquidityBucket"] = snapshot["AvgTradedValue20"].apply(_liquidity_bucket)
    snapshot["LiquidityEligible"] = snapshot["AvgTradedValue20"].fillna(0) >= LIQUIDITY_THRESHOLD
    snapshot["DataQualityStatus"] = snapshot["HistoryEligible"].map({True: "OK", False: "Insufficient history"})
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
    """
    Detect active setups first and rank only those setups.

    Trend state is context, not a buy signal by itself. A stock receives an
    active setup only when a current or recent trigger exists.
    """
    if snapshot.empty:
        return pd.DataFrame()

    df = snapshot.copy()

    # Safe numeric/boolean inputs. Existing public columns are preserved.
    rs3 = pd.to_numeric(df.get("RS3MPct"), errors="coerce").fillna(0)
    rs6 = pd.to_numeric(df.get("RS6MPct"), errors="coerce").fillna(0)
    volume = pd.to_numeric(df.get("VolumeRatio"), errors="coerce").fillna(0)
    atr_pct = pd.to_numeric(df.get("ATRPercent"), errors="coerce")
    days_momentum = pd.to_numeric(df.get("DaysSince9_21"), errors="coerce")
    days_swing = pd.to_numeric(df.get("DaysSince20_50"), errors="coerce")

    bull_regime = df.get("BullRegime", False)
    bull_swing = df.get("BullSwing", False)
    bull_momentum = df.get("BullMomentum", False)
    momentum_fresh = df.get("MomentumFresh", False)
    swing_fresh = df.get("SwingFresh", False)
    pullback = df.get("Pullback", False)
    breakout = df.get("Breakout20", False)
    volume_confirmed = df.get("VolumeConfirmedMomentum", False)

    for name, series in {
        "BullRegime": bull_regime,
        "BullSwing": bull_swing,
        "BullMomentum": bull_momentum,
        "MomentumFresh": momentum_fresh,
        "SwingFresh": swing_fresh,
        "Pullback": pullback,
        "Breakout20": breakout,
        "VolumeConfirmedMomentum": volume_confirmed,
    }.items():
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=df.index)
        df[name] = series.fillna(False).astype(bool)

    rsi_cols = [col for col in df.columns if re.match(r"^RSI\\d+$", str(col))]
    rsi = (
        pd.to_numeric(df[rsi_cols[0]], errors="coerce")
        if rsi_cols
        else pd.Series(float("nan"), index=df.index)
    )

    # Baseline fields are retained for compatibility with existing pages.
    df["RegimeScore"] = df["BullRegime"].astype(int) * 25
    df["MomentumScore"] = (
        df["BullMomentum"].astype(int) * 20
        + df["MomentumFresh"].astype(int) * 5
    )
    df["SwingScore"] = (
        df["BullSwing"].astype(int) * 20
        + df["SwingFresh"].astype(int) * 5
    )
    near_ema = pd.to_numeric(
        df.get("EMA255DistancePct"), errors="coerce"
    ).abs() <= 2
    df["EntryScore"] = (
        near_ema.astype(int) * 15
        + ((rsi >= 35) & (rsi <= 60)).astype(int) * 10
    )
    df["TrendScore"] = (
        df["RegimeScore"]
        + df["MomentumScore"]
        + df["SwingScore"]
    )

    strong_trend = (
        df["BullRegime"]
        & df["BullSwing"]
        & df["BullMomentum"]
    )

    # ------------------------------------------------------------
    # 1. Fresh Momentum
    # ------------------------------------------------------------
    recent_momentum_cross = (
        df["MomentumFresh"]
        | (days_momentum.notna() & (days_momentum <= 5))
    )
    fresh_momentum_active = (
        recent_momentum_cross
        & df["BullMomentum"]
        & (df["BullRegime"] | df["BullSwing"])
    )
    fresh_momentum_score = (
        df["BullRegime"].astype(int) * 20
        + df["BullSwing"].astype(int) * 15
        + df["BullMomentum"].astype(int) * 15
        + recent_momentum_cross.astype(int) * 25
        + (volume >= 1.5).astype(int) * 10
        + (rs3 >= 70).astype(int) * 10
        + (rsi >= 50).astype(int) * 5
    )
    df["FreshMomentumScore"] = 0
    df.loc[fresh_momentum_active, "FreshMomentumScore"] = (
        fresh_momentum_score.loc[fresh_momentum_active]
    )

    # ------------------------------------------------------------
    # 2. Pullback in Bull Regime
    # ------------------------------------------------------------
    pullback_active = (
        df["Pullback"]
        & df["BullRegime"]
        & df["BullSwing"]
    )
    pullback_score = (
        df["BullRegime"].astype(int) * 25
        + df["BullSwing"].astype(int) * 20
        + df["Pullback"].astype(int) * 30
        + (rs3 >= 50).astype(int) * 15
        + (volume >= 0.8).astype(int) * 5
        + (atr_pct <= 6).fillna(False).astype(int) * 5
    )
    df["PullbackScore"] = 0
    df.loc[pullback_active, "PullbackScore"] = (
        pullback_score.loc[pullback_active]
    )

    # ------------------------------------------------------------
    # 3. Volume-confirmed Breakout
    # ------------------------------------------------------------
    breakout_active = (
        df["Breakout20"]
        & df["BullRegime"]
    )
    breakout_score = (
        df["BullRegime"].astype(int) * 20
        + df["BullSwing"].astype(int) * 15
        + df["Breakout20"].astype(int) * 30
        + (volume >= 1.5).astype(int) * 15
        + (rs3 >= 70).astype(int) * 15
        + (rsi >= 50).astype(int) * 5
    )
    df["BreakoutScore"] = 0
    df.loc[breakout_active, "BreakoutScore"] = (
        breakout_score.loc[breakout_active]
    )

    # ------------------------------------------------------------
    # 4. Trend Continuation
    #
    # Strong trend alone is deliberately insufficient. Require a
    # continuation event: a recent 20/50 cross or current
    # volume-confirmed positive momentum. Breakouts have their own path.
    # ------------------------------------------------------------
    recent_swing_cross = (
        df["SwingFresh"]
        | (days_swing.notna() & (days_swing <= 10))
    )
    continuation_trigger = (
        recent_swing_cross
        | df["VolumeConfirmedMomentum"]
    )
    trend_continuation_active = (
        strong_trend
        & continuation_trigger
        & ~df["Breakout20"]
        & (rs3 >= 60)
    )
    trend_score = (
        df["BullRegime"].astype(int) * 20
        + df["BullSwing"].astype(int) * 20
        + df["BullMomentum"].astype(int) * 15
        + recent_swing_cross.astype(int) * 20
        + df["VolumeConfirmedMomentum"].astype(int) * 10
        + (rs3 >= 70).astype(int) * 10
        + (rs6 >= 60).astype(int) * 5
    )
    df["TrendContinuationScore"] = 0
    df.loc[trend_continuation_active, "TrendContinuationScore"] = (
        trend_score.loc[trend_continuation_active]
    )

    score_cols = [
        "TrendContinuationScore",
        "PullbackScore",
        "FreshMomentumScore",
        "BreakoutScore",
    ]
    setup_labels = {
        "TrendContinuationScore": "Trend Continuation",
        "PullbackScore": "Pullback in Bull Regime",
        "FreshMomentumScore": "Fresh Momentum",
        "BreakoutScore": "Volume Breakout",
    }

    df["SetupScore"] = df[score_cols].max(axis=1)
    best_col = df[score_cols].idxmax(axis=1)
    df["Setup"] = best_col.map(setup_labels)
    df.loc[df["SetupScore"] <= 0, "Setup"] = "No active setup"

    # Public ranking field used by the existing UI.
    df["ConvergenceScore"] = df["SetupScore"].round(0).astype(int)

    return df.sort_values(
        ["ConvergenceScore", "RS3MPct", "AvgTradedValue20"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

def investor_quality_gate(convergence: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a stricter retail-investor quality gate to Confluence candidates.

    This is a second-stage filter. Confluence remains intentionally broader.
    No fixed number of stocks is selected.

    The gate is setup-aware:
      - Trend Continuation already has its strict path-specific rules.
      - Fresh Momentum requires broader trend support, healthy RSI and participation.
      - Breakout requires a true breakout, volume and relative strength.
      - Pullback requires the existing EMA255/RSI trigger plus bullish structure.

    Returns candidates ranked by InvestorTechnicalScore.
    """
    if convergence.empty:
        return pd.DataFrame()

    df = convergence.copy()

    def num(col, default=0.0):
        return pd.to_numeric(
            df.get(col, pd.Series(default, index=df.index)),
            errors="coerce",
        )

    def flag(col):
        return df.get(
            col,
            pd.Series(False, index=df.index),
        ).fillna(False).astype(bool)

    rs3 = num("RS3MPct")
    rs6 = num("RS6MPct")
    volume = num("VolumeRatio")
    atr = num("ATRPercent")
    gap = num("GapPct")
    rsi = num("RSI14")

    history = df.get(
        "HistoryEligible",
        pd.Series(False, index=df.index),
    ).fillna(False).astype(bool)

    active = (
        df.get("Setup", pd.Series("No active setup", index=df.index)).astype(str).ne("No active setup")
        & (num("ConvergenceScore") >= 70)
        & history
    )

    common_quality = (
        (rs3 >= 75)
        & (atr <= 6)
        & (gap.abs() <= 5)
    )

    setup = df.get(
        "Setup",
        pd.Series("No active setup", index=df.index),
    ).astype(str)

    # Setup-specific quality gates.
    trend_ok = (
        (setup == "Trend Continuation")
        & flag("BullRegime")
        & flag("BullSwing")
        & flag("BullMomentum")
        & (rs3 >= 85)
        & (volume >= 1.5)
        & rsi.between(55, 75, inclusive="both")
    )

    momentum_ok = (
        (setup == "Fresh Momentum")
        & flag("MomentumFresh")
        & flag("BullMomentum")
        & (flag("BullRegime") | flag("BullSwing"))
        & (rs3 >= 75)
        & (volume >= 1.0)
        & rsi.between(50, 75, inclusive="both")
    )

    breakout_ok = (
        (setup == "Volume Breakout")
        & flag("Breakout20")
        & flag("BullRegime")
        & (rs3 >= 75)
        & (volume >= 1.5)
        & rsi.between(50, 80, inclusive="both")
    )

    pullback_ok = (
        (setup == "Pullback in Bull Regime")
        & flag("Pullback")
        & flag("BullRegime")
        & flag("BullSwing")
        & (rs3 >= 70)
        & rsi.lt(35)
        & (num("EMA255DistancePct").abs() <= 2)
    )

    candidate = active & common_quality & (
        trend_ok | momentum_ok | breakout_ok | pullback_ok
    )

    # Quality score is for ranking valid candidates, not creating signals.
    setup_component = num("ConvergenceScore").clip(0, 100) * 0.35
    rs_component = rs3.clip(0, 100) * 0.20
    rs6_component = rs6.clip(0, 100) * 0.10
    volume_component = (
        volume.clip(lower=0, upper=2.5) / 2.5
    ) * 10

    trend_component = (
        (
            flag("BullRegime").astype(int)
            + flag("BullSwing").astype(int)
            + flag("BullMomentum").astype(int)
        ) / 3
    ) * 10

    # Entry quality rewards a useful RSI zone without forcing every setup
    # into the same RSI behavior.
    entry_component = pd.Series(0.0, index=df.index)
    entry_component = entry_component.mask(
        setup.eq("Fresh Momentum"),
        rsi.between(50, 75, inclusive="both").astype(int) * 10,
    )
    entry_component = entry_component.mask(
        setup.eq("Volume Breakout"),
        rsi.between(50, 80, inclusive="both").astype(int) * 10,
    )
    entry_component = entry_component.mask(
        setup.eq("Trend Continuation"),
        rsi.between(55, 75, inclusive="both").astype(int) * 10,
    )
    entry_component = entry_component.mask(
        setup.eq("Pullback in Bull Regime"),
        rsi.lt(35).astype(int) * 10,
    )

    risk_component = (
        (atr <= 4).astype(int) * 3
        + ((atr > 4) & (atr <= 6)).astype(int) * 2
        + (gap.abs() <= 2).astype(int) * 2
    )

    df["InvestorTechnicalScore"] = (
        setup_component
        + rs_component
        + rs6_component
        + volume_component
        + trend_component
        + entry_component
        + risk_component
    ).round(1)

    df["InvestorQualityPass"] = candidate
    df.loc[~candidate, "InvestorTechnicalScore"] = 0.0

    return (
        df.loc[candidate]
        .sort_values(
            ["InvestorTechnicalScore", "ConvergenceScore", "RS3MPct"],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

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
