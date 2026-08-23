
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
TOTAL_MARKET_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarketlist.csv",
]

NIFTY500_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]

MICROCAP250_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250list.csv",
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

    # Try the direct Total Market constituent file first.
    try:
        direct = _read_nse_csv(TOTAL_MARKET_URLS)
        if len(direct) >= 700:
            return direct
    except Exception:
        pass

    # Robust fallback. NSE states Total Market is Nifty 500 + Microcap 250.
    nifty500 = _read_nse_csv(NIFTY500_URLS)
    microcap = _read_nse_csv(MICROCAP250_URLS)

    result = (
        pd.concat([nifty500, microcap], ignore_index=True)
        .drop_duplicates("Symbol")
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


def calculate_indicators(
    prices: pd.DataFrame,
    ema_long: int = 255,
    rsi_period: int = 14,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    result_frames = []

    for ticker, frame in prices.groupby(
        "Yahoo Symbol",
        sort=False,
    ):
        frame = (
            frame
            .sort_values("Date")
            .drop_duplicates("Date")
            .copy()
        )

        close = frame["Close"]

        frame["EMA9"] = close.ewm(span=9, adjust=False).mean()
        frame["EMA21"] = close.ewm(span=21, adjust=False).mean()

        frame["SMA20"] = close.rolling(
            20,
            min_periods=20,
        ).mean()

        frame["SMA50"] = close.rolling(
            50,
            min_periods=50,
        ).mean()

        frame["SMA200"] = close.rolling(
            200,
            min_periods=200,
        ).mean()

        frame[f"EMA{ema_long}"] = close.ewm(
            span=ema_long,
            adjust=False,
            min_periods=ema_long,
        ).mean()

        frame[f"RSI{rsi_period}"] = rsi_wilder(
            close,
            rsi_period,
        )

        # Cross states. These use yesterday vs today, avoiding look-ahead.
        frame["Cross9_21"] = (
            (frame["EMA9"] > frame["EMA21"])
            & (frame["EMA9"].shift(1) <= frame["EMA21"].shift(1))
        )

        frame["Cross20_50"] = (
            (frame["SMA20"] > frame["SMA50"])
            & (frame["SMA20"].shift(1) <= frame["SMA50"].shift(1))
        )

        frame["Cross50_200"] = (
            (frame["SMA50"] > frame["SMA200"])
            & (frame["SMA50"].shift(1) <= frame["SMA200"].shift(1))
        )

        frame["BullMomentum"] = frame["EMA9"] > frame["EMA21"]
        frame["BullSwing"] = frame["SMA20"] > frame["SMA50"]
        frame["BullRegime"] = frame["SMA50"] > frame["SMA200"]

        frame["EMA255DistancePct"] = (
            (frame["Close"] - frame[f"EMA{ema_long}"])
            / frame[f"EMA{ema_long}"]
            * 100
        )

        frame["Pullback"] = (
            (frame[f"RSI{rsi_period}"] < 35)
            & (frame["EMA255DistancePct"].abs() <= 2)
        )

        frame["MomentumFresh"] = frame["Cross9_21"]
        frame["SwingFresh"] = frame["Cross20_50"]
        frame["RegimeFresh"] = frame["Cross50_200"]

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

    meta = (
        universe
        .set_index("Yahoo Symbol")[["Symbol", "Company"]]
        .to_dict("index")
    )

    rows = []

    for ticker, frame in indicators.groupby(
        "Yahoo Symbol",
        sort=False,
    ):
        row = frame.sort_values("Date").iloc[-1]
        company = meta.get(
            ticker,
            {
                "Symbol": ticker.replace(".NS", ""),
                "Company": "",
            },
        )

        rows.append(
            {
                "Symbol": company["Symbol"],
                "Company": company["Company"],
                "Yahoo Symbol": ticker,
                "Date": row["Date"],
                "Close": row["Close"],
                "EMA9": row["EMA9"],
                "EMA21": row["EMA21"],
                "SMA20": row["SMA20"],
                "SMA50": row["SMA50"],
                "SMA200": row["SMA200"],
                f"EMA{ema_long}": row[f"EMA{ema_long}"],
                f"RSI{rsi_period}": row[f"RSI{rsi_period}"],
                "EMA255DistancePct": row["EMA255DistancePct"],
                "BullMomentum": bool(row["BullMomentum"]),
                "BullSwing": bool(row["BullSwing"]),
                "BullRegime": bool(row["BullRegime"]),
                "MomentumFresh": bool(row["MomentumFresh"]),
                "SwingFresh": bool(row["SwingFresh"]),
                "RegimeFresh": bool(row["RegimeFresh"]),
                "Pullback": bool(row["Pullback"]),
            }
        )

    return pd.DataFrame(rows)


def add_days_since_cross(
    indicators: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Days since the latest bullish crossover for each stock."""
    if indicators.empty or snapshot.empty:
        return snapshot

    cross_dates = {}

    for ticker, frame in indicators.groupby(
        "Yahoo Symbol",
        sort=False,
    ):
        latest_date = frame["Date"].max()

        latest_9 = frame.loc[
            frame["Cross9_21"] & (frame["Date"] <= latest_date),
            "Date",
        ]
        latest_20 = frame.loc[
            frame["Cross20_50"] & (frame["Date"] <= latest_date),
            "Date",
        ]
        latest_50 = frame.loc[
            frame["Cross50_200"] & (frame["Date"] <= latest_date),
            "Date",
        ]

        cross_dates[ticker] = {
            "DaysSince9_21": (
                int((latest_date - latest_9.iloc[-1]).days)
                if not latest_9.empty
                else None
            ),
            "DaysSince20_50": (
                int((latest_date - latest_20.iloc[-1]).days)
                if not latest_20.empty
                else None
            ),
            "DaysSince50_200": (
                int((latest_date - latest_50.iloc[-1]).days)
                if not latest_50.empty
                else None
            ),
        }

    extra = pd.DataFrame.from_dict(
        cross_dates,
        orient="index",
    )
    extra.index.name = "Yahoo Symbol"

    return snapshot.join(
        extra,
        on="Yahoo Symbol",
    )


def convergence_table(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame()

    df = snapshot.copy()

    # Do not treat the related moving-average states as independent signals.
    # Instead we score three trend dimensions plus entry/pullback.
    df["RegimeScore"] = (
        df["BullRegime"].astype(int) * 25
    )
    df["MomentumScore"] = (
        df["BullMomentum"].astype(int) * 20
        + df["MomentumFresh"].astype(int) * 5
    )
    df["SwingScore"] = (
        df["BullSwing"].astype(int) * 20
        + df["SwingFresh"].astype(int) * 5
    )

    # Entry score is deliberately independent of trend direction.
    rsi = df.filter(regex=r"^RSI\d+$").iloc[:, 0]
    near_ema = df["EMA255DistancePct"].abs() <= 2

    entry = (
        (near_ema.astype(int) * 15)
        + ((rsi >= 35) & (rsi <= 60)).astype(int) * 10
    )

    # Oversold pullback is a setup, not automatically a positive score.
    pullback_flag = df["Pullback"].astype(int)

    df["EntryScore"] = entry
    df["TrendScore"] = (
        df["RegimeScore"]
        + df["MomentumScore"]
        + df["SwingScore"]
    )
    df["ConvergenceScore"] = df["TrendScore"] + df["EntryScore"]

    df["Setup"] = "No setup"
    df.loc[pullback_flag.eq(1) & df["BullRegime"], "Setup"] = (
        "Pullback in Bull Regime"
    )
    df.loc[
        df["MomentumFresh"]
        & df["BullSwing"]
        & df["BullRegime"],
        "Setup",
    ] = "Fresh Momentum"
    df.loc[
        df["SwingFresh"]
        & df["BullRegime"],
        "Setup",
    ] = "Fresh Swing"
    df.loc[
        (df["TrendScore"] >= 60)
        & (df["EntryScore"] >= 15),
        "Setup",
    ] = "Trend + Entry"

    return df.sort_values(
        ["ConvergenceScore", "TrendScore", "EntryScore"],
        ascending=False,
    ).reset_index(drop=True)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fundamental_snapshot(ticker: str) -> dict:
    """Fetch fundamentals only after a stock becomes a finalist."""
    defaults = {
        "PE": None,
        "PB": None,
        "Profit Margin %": None,
        "Debt/Equity": None,
        "EV/EBITDA": None,
        "Market Cap": None,
    }

    try:
        info = yf.Ticker(ticker).info or {}

        profit = info.get("profitMargins")
        return {
            "PE": info.get("trailingPE"),
            "PB": info.get("priceToBook"),
            "Profit Margin %": (
                profit * 100 if profit is not None else None
            ),
            "Debt/Equity": info.get("debtToEquity"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "Market Cap": info.get("marketCap"),
        }
    except Exception:
        return defaults
