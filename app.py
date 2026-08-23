import io
import time
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Nifty 500 EMA Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NIFTY500_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
)


@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty500_constituents():
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
    }
    errors = []

    for url in NIFTY500_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))

            columns = {
                str(column).strip().lower(): column
                for column in df.columns
            }

            symbol_col = columns.get("symbol")
            company_col = (
                columns.get("company name")
                or columns.get("company")
                or columns.get("name")
            )

            if symbol_col is None or company_col is None:
                raise ValueError(
                    f"Unexpected columns: {df.columns.tolist()}"
                )

            result = (
                df[[symbol_col, company_col]]
                .rename(
                    columns={
                        symbol_col: "Symbol",
                        company_col: "Company",
                    }
                )
                .dropna()
                .assign(
                    Symbol=lambda x: x["Symbol"]
                    .astype(str)
                    .str.strip()
                )
                .drop_duplicates(subset=["Symbol"])
                .sort_values("Symbol")
                .reset_index(drop=True)
            )

            if len(result) < 400:
                raise ValueError(
                    f"Only {len(result)} constituents returned."
                )

            result["Yahoo Symbol"] = result["Symbol"] + ".NS"
            return result

        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to download Nifty 500 constituents.\n"
        + "\n".join(errors)
    )


def normalize_download(data, tickers):
    frames = []

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                frame = data[ticker].copy()
            except (KeyError, TypeError):
                continue

            frame = frame.dropna(how="all")

            if (
                not frame.empty
                and {"Close", "High", "Low"}.issubset(frame.columns)
            ):
                frame = frame.reset_index()
                frame["Yahoo Symbol"] = ticker
                frames.append(frame)

    elif len(tickers) == 1:
        frame = data.dropna(how="all").copy()

        if (
            not frame.empty
            and {"Close", "High", "Low"}.issubset(frame.columns)
        ):
            frame = frame.reset_index()
            frame["Yahoo Symbol"] = tickers[0]
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result.dropna(
        subset=["Date", "High", "Low", "Close"]
    ).reset_index(drop=True)


@st.cache_data(ttl=14400, show_spinner=False)
def download_price_batch(tickers_tuple, start_date, end_date):
    tickers = list(tickers_tuple)

    for attempt in range(3):
        try:
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )

            result = normalize_download(data, tickers)

            if not result.empty:
                return result

        except Exception:
            pass

        time.sleep(1.5 * (attempt + 1))

    return pd.DataFrame()


def download_all_prices(
    tickers,
    start_date,
    end_date,
    batch_size,
    progress_callback=None,
):
    all_frames = []
    failures = []

    total_batches = (
        len(tickers) + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(0, len(tickers), batch_size),
        start=1,
    ):
        batch = list(tickers[start:start + batch_size])

        frame = download_price_batch(
            tuple(batch),
            start_date,
            end_date,
        )

        if not frame.empty:
            all_frames.append(frame)
            returned = set(frame["Yahoo Symbol"].unique())
            missing = [
                ticker
                for ticker in batch
                if ticker not in returned
            ]
        else:
            missing = batch

        # Retry missing tickers individually.
        for ticker in missing:
            retry = download_price_batch(
                (ticker,),
                start_date,
                end_date,
            )

            if not retry.empty:
                all_frames.append(retry)
            else:
                failures.append(ticker)

        if progress_callback:
            progress_callback(
                batch_number,
                total_batches,
                len(failures),
            )

        time.sleep(0.25)

    if not all_frames:
        return pd.DataFrame(), sorted(set(failures))

    prices = (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates(
            subset=["Date", "Yahoo Symbol"],
            keep="last",
        )
        .sort_values(["Yahoo Symbol", "Date"])
        .reset_index(drop=True)
    )

    return prices, sorted(set(failures))



def calculate_rsi(close, period=14):
    """Calculate Wilder's RSI."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def get_latest_buy_candidates(
    universe,
    prices,
    ema_period,
    rsi_period,
    rsi_threshold=35.0,
    ema_distance_limit=2.0,
):
    """
    Build a current buying list using the latest available trading day.

    Conditions:
    1. RSI < rsi_threshold
    2. Distance from EMA is between -ema_distance_limit% and
       +ema_distance_limit%
    """
    if prices.empty:
        return pd.DataFrame()

    company_lookup = (
        universe
        .set_index("Yahoo Symbol")[["Symbol", "Company"]]
        .to_dict("index")
    )

    candidates = []

    for ticker, frame in prices.groupby("Yahoo Symbol", sort=False):
        frame = (
            frame
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
            .copy()
        )

        if len(frame) < max(ema_period, rsi_period):
            continue

        frame["EMA"] = frame["Close"].ewm(
            span=ema_period,
            adjust=False,
            min_periods=ema_period,
        ).mean()

        frame["RSI"] = calculate_rsi(
            frame["Close"],
            period=rsi_period,
        )

        latest = frame.iloc[-1]

        if pd.isna(latest["EMA"]) or pd.isna(latest["RSI"]):
            continue

        distance_pct = (
            (
                float(latest["Close"])
                - float(latest["EMA"])
            )
            / float(latest["EMA"])
            * 100
        )

        if (
            float(latest["RSI"]) < rsi_threshold
            and abs(distance_pct) <= ema_distance_limit
        ):
            metadata = company_lookup.get(
                ticker,
                {
                    "Symbol": ticker.replace(".NS", ""),
                    "Company": "",
                },
            )

            candidates.append(
                {
                    "Symbol": metadata["Symbol"],
                    "Company": metadata["Company"],
                    "Yahoo Symbol": ticker,
                    "RSI": float(latest["RSI"]),
                    "Distance from EMA %": distance_pct,
                    "Close": float(latest["Close"]),
                }
            )

    if not candidates:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Company",
                "Yahoo Symbol",
                "RSI",
                "Distance from EMA %",
                "Close",
            ]
        )

    return (
        pd.DataFrame(candidates)
        .sort_values(
            ["RSI", "Distance from EMA %"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )


@st.cache_data(ttl=21600, show_spinner=False)
def get_stock_fundamentals(ticker):
    """Fetch the six Buying List fundamentals from free Yahoo data."""
    empty = {
        "Trailing P/E": None,
        "P/B": None,
        "Profit Margin %": None,
        "Debt/Equity %": None,
        "EV/EBITDA": None,
        "Market Cap": None,
    }

    try:
        info = yf.Ticker(ticker).info or {}

        profit_margin = info.get("profitMargins")
        debt_equity = info.get("debtToEquity")

        return {
            "Trailing P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "Profit Margin %": (
                float(profit_margin) * 100
                if profit_margin is not None
                else None
            ),
            "Debt/Equity %": (
                float(debt_equity)
                if debt_equity is not None
                else None
            ),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "Market Cap": info.get("marketCap"),
        }
    except Exception:
        return empty


def format_market_cap(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if value >= 1e12:
        return f"₹{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"₹{value / 1e9:.2f}B"
    if value >= 1e7:
        return f"₹{value / 1e7:.2f}Cr"

    return f"₹{value:,.0f}"


def add_fundamentals_to_buying_list(candidates):
    """Add the final six fundamentals to current technical candidates."""
    if candidates.empty:
        return candidates

    rows = []

    for _, row in candidates.iterrows():
        fundamentals = get_stock_fundamentals(row["Yahoo Symbol"])

        rows.append(
            {
                "Symbol": row["Symbol"],
                "Company": row["Company"],
                "Yahoo Symbol": row["Yahoo Symbol"],
                "Close": round(float(row["Close"]), 2),
                "RSI": round(float(row["RSI"]), 2),
                "Distance from EMA %": round(
                    float(row["Distance from EMA %"]),
                    2,
                ),
                "Trailing P/E": fundamentals["Trailing P/E"],
                "P/B": fundamentals["P/B"],
                "Profit Margin %": fundamentals["Profit Margin %"],
                "Debt/Equity %": fundamentals["Debt/Equity %"],
                "EV/EBITDA": fundamentals["EV/EBITDA"],
                "Market Cap": fundamentals["Market Cap"],
            }
        )

    output = pd.DataFrame(rows)

    numeric_columns = [
        "Trailing P/E",
        "P/B",
        "Profit Margin %",
        "Debt/Equity %",
        "EV/EBITDA",
    ]

    for column in numeric_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).round(2)

    output["Market Cap"] = output["Market Cap"].apply(
        format_market_cap
    )

    output.insert(
        0,
        "Rank",
        range(1, len(output) + 1),
    )

    return output


def build_buying_chart(frame, ema_period, rsi_period, chart_days):
    """Create a Bloomberg-style price, EMA and RSI chart."""
    chart_frame = (
        frame
        .sort_values("Date")
        .copy()
    )

    chart_frame["EMA"] = chart_frame["Close"].ewm(
        span=ema_period,
        adjust=False,
        min_periods=ema_period,
    ).mean()

    chart_frame["RSI"] = calculate_rsi(
        chart_frame["Close"],
        period=rsi_period,
    )

    chart_frame = chart_frame.tail(chart_days)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_frame["Date"],
            open=chart_frame["Open"],
            high=chart_frame["High"],
            low=chart_frame["Low"],
            close=chart_frame["Close"],
            name="Price",
            increasing_line_color="#2dd4bf",
            decreasing_line_color="#fb7185",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_frame["Date"],
            y=chart_frame["EMA"],
            mode="lines",
            name=f"EMA {ema_period}",
            line={"color": "#f59e0b", "width": 2},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_frame["Date"],
            y=chart_frame["RSI"],
            mode="lines",
            name=f"RSI {rsi_period}",
            line={"color": "#60a5fa", "width": 2},
        ),
        row=2,
        col=1,
    )

    for level in [30, 35, 70]:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color="#475569",
            row=2,
            col=1,
        )

    fig.update_layout(
        height=680,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        paper_bgcolor="#080c12",
        plot_bgcolor="#080c12",
        font={"color": "#d8e0ea", "family": "Arial"},
        legend={
            "orientation": "h",
            "y": 1.04,
            "x": 0,
        },
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="#1f2937",
        showline=False,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#1f2937",
        showline=False,
        zeroline=False,
    )

    fig.update_yaxes(
        range=[0, 100],
        row=2,
        col=1,
    )

    return fig



def scan_prices(
    universe,
    prices,
    ema_period,
    touch_mode,
    tolerance_pct,
    rsi_period,
):
    """Calculate EMA and retain the latest qualifying touch per stock."""

    if prices.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Symbol",
                "Company",
                "Signal Type",
                "Occurrence Date",
                "Close",
                f"EMA {ema_period}",
                f"RSI {rsi_period}",
                "RSI Zone",
                "Distance from EMA %",
            ]
        )

    company_lookup = (
        universe
        .set_index("Yahoo Symbol")[["Symbol", "Company"]]
        .to_dict("index")
    )

    results = []

    for ticker, frame in prices.groupby(
        "Yahoo Symbol",
        sort=False,
    ):
        frame = (
            frame
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
            .copy()
        )

        if len(frame) < ema_period:
            continue

        frame["EMA"] = frame["Close"].ewm(
            span=ema_period,
            adjust=False,
            min_periods=ema_period,
        ).mean()

        frame["RSI"] = calculate_rsi(
            frame["Close"],
            period=rsi_period,
        )

        # Keep previous values in THIS DataFrame.
        frame["Previous Close"] = frame["Close"].shift(1)
        frame["Previous EMA"] = frame["EMA"].shift(1)

        if touch_mode == "Wick":
            touches = (
                frame["Low"].le(frame["EMA"])
                & frame["High"].ge(frame["EMA"])
                & frame["EMA"].notna()
            )
        else:
            distance = (
                (frame["Close"] - frame["EMA"]).abs()
                / frame["EMA"].abs()
            )

            touches = (
                distance.le(tolerance_pct / 100)
                & frame["EMA"].notna()
            )

        hits = frame.loc[touches]

        if hits.empty:
            continue

        hit = hits.iloc[-1]

        previous_close = hit["Previous Close"]
        previous_ema = hit["Previous EMA"]

        close_distance = (
            abs(
                float(hit["Close"])
                - float(hit["EMA"])
            )
            / max(
                abs(float(hit["EMA"])),
                1e-12,
            )
        )

        if close_distance <= 0.001:
            signal_type = "Close on EMA"
        elif (
            pd.notna(previous_close)
            and pd.notna(previous_ema)
        ):
            if previous_close > previous_ema:
                signal_type = "Touched from Above"
            elif previous_close < previous_ema:
                signal_type = "Touched from Below"
            else:
                signal_type = "Touch"
        else:
            signal_type = "Touch"

        rsi_value = (
            float(hit["RSI"])
            if pd.notna(hit["RSI"])
            else None
        )

        if rsi_value is None:
            rsi_zone = "N/A"
        elif rsi_value >= 70:
            rsi_zone = "Overbought"
        elif rsi_value <= 30:
            rsi_zone = "Oversold"
        else:
            rsi_zone = "Neutral"

        metadata = company_lookup.get(
            ticker,
            {
                "Symbol": ticker.replace(".NS", ""),
                "Company": "",
            },
        )

        results.append(
            {
                "Symbol": metadata["Symbol"],
                "Company": metadata["Company"],
                "Signal Type": signal_type,
                "Occurrence Date": hit["Date"],
                "Close": float(hit["Close"]),
                f"EMA {ema_period}": float(hit["EMA"]),
                f"RSI {rsi_period}": rsi_value,
                "RSI Zone": rsi_zone,
                "Distance from EMA %": (
                    (
                        float(hit["Close"])
                        - float(hit["EMA"])
                    )
                    / float(hit["EMA"])
                    * 100
                ),
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Symbol",
                "Company",
                "Signal Type",
                "Occurrence Date",
                "Close",
                f"EMA {ema_period}",
                "Distance from EMA %",
            ]
        )

    output = pd.DataFrame(results)

    output["Occurrence Date"] = pd.to_datetime(
        output["Occurrence Date"]
    )

    output = output.sort_values(
        ["Occurrence Date", "Symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)

    output.insert(
        0,
        "Rank",
        range(1, len(output) + 1),
    )

    output["Close"] = output["Close"].round(2)
    output[f"EMA {ema_period}"] = (
        output[f"EMA {ema_period}"].round(2)
    )
    output[f"RSI {rsi_period}"] = (
        output[f"RSI {rsi_period}"].round(2)
    )
    output["Distance from EMA %"] = (
        output["Distance from EMA %"].round(2)
    )

    output["Occurrence Date"] = (
        output["Occurrence Date"]
        .dt.strftime("%Y-%m-%d")
    )

    return output




# ---------------- UI ----------------

st.markdown(
    """
<style>
    .stApp {
        background: #080c12;
        color: #d8e0ea;
    }
    [data-testid="stSidebar"] {
        background: #0d131c;
        border-right: 1px solid #253243;
    }
    .terminal-header {
        border: 1px solid #253243;
        border-left: 4px solid #f59e0b;
        background: #0d131c;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .terminal-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.6px;
    }
    .terminal-subtitle {
        color: #8fa3b8;
        font-size: 13px;
        margin-top: 4px;
        font-family: monospace;
    }
    .terminal-label {
        color: #f59e0b;
        font-family: monospace;
        font-size: 12px;
        letter-spacing: 1px;
    }
    .stMetric {
        background: #0d131c;
        border: 1px solid #253243;
        border-top: 2px solid #f59e0b;
        padding: 10px 14px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #253243;
    }
    .stButton > button {
        border-radius: 2px;
        border: 1px solid #f59e0b;
        background: #121a25;
        color: #f8fafc;
        font-weight: 700;
    }
    .stButton > button:hover {
        background: #f59e0b;
        color: #080c12;
    }
    .stDownloadButton > button {
        border-radius: 2px;
        border: 1px solid #334155;
        background: #121a25;
        color: #d8e0ea;
    }
    h1, h2, h3 {
        letter-spacing: 0.3px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="terminal-header">
    <div class="terminal-label">NSE / EQUITY SCREENER / TECHNICAL + FUNDAMENTAL</div>
    <div class="terminal-title">NIFTY 500 EMA TERMINAL</div>
    <div class="terminal-subtitle">EMA 255 · RSI · CURRENT BUY CANDIDATES · FREE DATA</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scanner Settings")

    ema_period = st.number_input(
        "EMA period",
        min_value=2,
        max_value=1000,
        value=255,
        step=1,
    )

    history_years = st.slider(
        "Price history",
        min_value=2,
        max_value=8,
        value=4,
    )

    rsi_period = st.number_input(
        "RSI period",
        min_value=2,
        max_value=100,
        value=14,
        step=1,
    )

    touch_label = st.radio(
        "Signal definition",
        [
            "Wick touches EMA",
            "Close near EMA",
        ],
    )

    tolerance_pct = st.number_input(
        "Close tolerance %",
        min_value=0.05,
        max_value=5.0,
        value=0.50,
        step=0.05,
        disabled=(
            touch_label == "Wick touches EMA"
        ),
    )

    batch_size = st.select_slider(
        "Download batch size",
        options=[25, 50, 75, 100],
        value=50,
    )

    st.divider()

    if st.button(
        "Clear cached market data",
        use_container_width=True,
    ):
        get_nifty500_constituents.clear()
        download_price_batch.clear()
        st.success("Market-data cache cleared.")

    run_scan = st.button(
        "Run Nifty 500 Scan",
        type="primary",
        use_container_width=True,
    )


if run_scan:
    try:
        progress_bar = st.progress(0.0)
        status = st.empty()

        with st.spinner(
            "Loading current Nifty 500 constituents..."
        ):
            universe = get_nifty500_constituents()

        status.info(
            f"Loaded {len(universe):,} constituents. "
            "Downloading daily price history..."
        )

        end_date = date.today() + timedelta(days=1)
        start_date = (
            end_date
            - timedelta(
                days=int(history_years * 365.25)
            )
        )

        def update_progress(batch, total, failures):
            progress_bar.progress(batch / total)
            status.info(
                f"Downloading batch {batch}/{total} "
                f"· unresolved tickers: {failures}"
            )

        prices, failures = download_all_prices(
            tickers=universe["Yahoo Symbol"].tolist(),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            batch_size=batch_size,
            progress_callback=update_progress,
        )

        if prices.empty:
            progress_bar.empty()
            status.error(
                "No usable historical price data was returned."
            )
            st.stop()

        progress_bar.progress(1.0)

        status.info(
            f"Downloaded "
            f"{prices['Yahoo Symbol'].nunique():,} stocks. "
            "Calculating EMA signals..."
        )

        touch_mode = (
            "Wick"
            if touch_label == "Wick touches EMA"
            else "Close"
        )

        with st.spinner(
            f"Calculating EMA {ema_period}..."
        ):
            signals = scan_prices(
                universe=universe,
                prices=prices,
                ema_period=int(ema_period),
                touch_mode=touch_mode,
                tolerance_pct=float(tolerance_pct),
                rsi_period=int(rsi_period),
            )

            buying_candidates = get_latest_buy_candidates(
                universe=universe,
                prices=prices,
                ema_period=int(ema_period),
                rsi_period=int(rsi_period),
                rsi_threshold=35.0,
                ema_distance_limit=2.0,
            )

            buying_list = add_fundamentals_to_buying_list(
                buying_candidates
            )

        st.session_state["signals"] = signals
        st.session_state["buying_list"] = buying_list
        st.session_state["prices"] = prices
        st.session_state["universe"] = universe
        st.session_state["ema_period"] = int(ema_period)
        st.session_state["scan_meta"] = {
            "constituents": len(universe),
            "downloaded": int(
                prices["Yahoo Symbol"].nunique()
            ),
            "failures": failures,
            "scan_date": date.today().isoformat(),
            "touch_label": touch_label,
            "history_years": history_years,
            "rsi_period": int(rsi_period),
        }

        progress_bar.empty()
        status.empty()

    except Exception as exc:
        st.error(
            "The scan could not be completed. "
            "Please try again in a few minutes."
        )
        with st.expander("Technical details"):
            st.exception(exc)


if "signals" not in st.session_state:
    st.info(
        "Configure the scanner in the sidebar and click "
        "**Run Nifty 500 Scan**."
    )
    st.stop()


signals = st.session_state["signals"]
current_ema = st.session_state["ema_period"]
meta = st.session_state["scan_meta"]

if meta["failures"]:
    st.warning(
        f"{len(meta['failures'])} ticker(s) could not be "
        "downloaded after retries."
    )

with st.expander("Scan details"):
    st.write(
        {
            "Scan date": meta["scan_date"],
            "Constituents loaded": meta["constituents"],
            "Stocks with price data": meta["downloaded"],
            "Failed tickers": len(meta["failures"]),
            "EMA period": current_ema,
            "RSI period": meta.get("rsi_period", 14),
            "Signal definition": meta["touch_label"],
            "History downloaded": (
                f"{meta['history_years']} years"
            ),
        }
    )

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric("Signals found", f"{len(signals):,}")

if signals.empty:
    metric_2.metric("Most recent signal", "None")
    metric_3.metric("Signals in last 7 days", "0")
else:
    metric_2.metric(
        "Most recent signal",
        signals["Occurrence Date"].iloc[0],
    )

    occurrence_dates = pd.to_datetime(
        signals["Occurrence Date"]
    )

    recent_count = int(
        (
            occurrence_dates
            >= pd.Timestamp.today().normalize()
            - pd.Timedelta(days=7)
        ).sum()
    )

    metric_3.metric(
        "Signals in last 7 days",
        recent_count,
    )

st.markdown(
    '<div class="terminal-label">WATCHLIST / BUYING CANDIDATES</div>',
    unsafe_allow_html=True,
)
st.subheader("Buying List")

buying_list = st.session_state.get(
    "buying_list",
    pd.DataFrame(),
)
stored_prices = st.session_state.get(
    "prices",
    pd.DataFrame(),
)

st.caption(
    "LIVE FILTER: RSI < 35  |  DISTANCE FROM EMA 255 ≤ ±2%"
)

if buying_list.empty:
    st.info(
        "No stocks currently meet both Buying List conditions."
    )
else:
    buy_col_1, buy_col_2, buy_col_3 = st.columns([1, 1, 2])

    buy_col_1.metric(
        "BUY CANDIDATES",
        f"{len(buying_list):,}",
    )
    buy_col_2.metric(
        "LOWEST RSI",
        f"{buying_list['RSI'].min():.2f}",
    )
    buy_col_3.metric(
        "FILTER",
        "RSI < 35 | EMA ±2%",
    )

    display_columns = [
        "Rank",
        "Symbol",
        "Company",
        "Close",
        "RSI",
        "Distance from EMA %",
        "Trailing P/E",
        "P/B",
        "Profit Margin %",
        "Debt/Equity %",
        "EV/EBITDA",
        "Market Cap",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in buying_list.columns
    ]

    st.dataframe(
        buying_list[display_columns],
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    buying_csv = buying_list[
        display_columns
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        "EXPORT BUYING LIST CSV",
        data=buying_csv,
        file_name="nifty500_buying_list.csv",
        mime="text/csv",
    )

    show_charts = st.toggle(
        "SHOW CHARTS FOR BUYING LIST STOCKS",
        value=False,
    )

    if show_charts:
        st.markdown(
            '<div class="terminal-label">CHART CONSOLE</div>',
            unsafe_allow_html=True,
        )

        chart_left, chart_right = st.columns([2, 1])

        with chart_left:
            selected_symbol = st.selectbox(
                "Select Buying List Stock",
                buying_list["Symbol"].tolist(),
            )

        with chart_right:
            chart_days = st.selectbox(
                "Chart Window",
                [90, 180, 252, 365],
                index=2,
            )

        selected_row = buying_list.loc[
            buying_list["Symbol"] == selected_symbol
        ].iloc[0]

        selected_ticker = selected_row["Yahoo Symbol"]

        chart_frame = stored_prices.loc[
            stored_prices["Yahoo Symbol"] == selected_ticker
        ].copy()

        if chart_frame.empty:
            st.warning(
                "Price history is not available for this stock."
            )
        else:
            st.markdown(
                f"### {selected_symbol}  |  {selected_row['Company']}"
            )

            chart_metrics = st.columns(4)
            chart_metrics[0].metric(
                "CLOSE",
                f"₹{selected_row['Close']:.2f}",
            )
            chart_metrics[1].metric(
                "RSI",
                f"{selected_row['RSI']:.2f}",
            )
            chart_metrics[2].metric(
                "EMA DISTANCE",
                f"{selected_row['Distance from EMA %']:.2f}%",
            )
            chart_metrics[3].metric(
                "MARKET CAP",
                selected_row["Market Cap"],
            )

            chart = build_buying_chart(
                frame=chart_frame,
                ema_period=current_ema,
                rsi_period=meta.get("rsi_period", 14),
                chart_days=int(chart_days),
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

st.divider()

st.markdown(
    '<div class="terminal-label">SIGNAL MONITOR / HISTORICAL TOUCHES</div>',
    unsafe_allow_html=True,
)
st.subheader("Ranked EMA Signals")

filter_col_1, filter_col_2 = st.columns([2, 1])

with filter_col_1:
    search = st.text_input(
        "Search symbol or company",
        placeholder="RELIANCE, TCS, HDFC...",
    )

with filter_col_2:
    max_days = st.number_input(
        "Only signals from last N days",
        min_value=1,
        max_value=3650,
        value=30,
        step=1,
    )

filtered = signals.copy()

if not filtered.empty:
    cutoff = (
        pd.Timestamp.today().normalize()
        - pd.Timedelta(days=int(max_days))
    )

    filtered = filtered[
        pd.to_datetime(
            filtered["Occurrence Date"]
        ) >= cutoff
    ].copy()

if search:
    mask = (
        filtered["Symbol"].str.contains(
            search,
            case=False,
            na=False,
        )
        | filtered["Company"].str.contains(
            search,
            case=False,
            na=False,
        )
    )

    filtered = filtered.loc[mask]

st.caption(
    f"Showing {len(filtered):,} "
    f"of {len(signals):,} detected signals."
)

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    height=600,
)

csv = signals.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download nifty500_ema_signals.csv",
    data=csv,
    file_name="nifty500_ema_signals.csv",
    mime="text/csv",
)

if meta["failures"]:
    failed_csv = (
        pd.DataFrame(
            {"Yahoo Symbol": meta["failures"]}
        )
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download failed tickers",
        data=failed_csv,
        file_name="nifty500_failed_tickers.csv",
        mime="text/csv",
    )

with st.expander("Signal methodology"):
    st.markdown(
        f"""
**Universe:** Current Nifty 500 constituents.

**EMA:** EMA calculated using **{current_ema}** daily closing prices.

**RSI:** Wilder's RSI using the selected period. RSI >= 70 is **Overbought**,
RSI <= 30 is **Oversold**, otherwise it is **Neutral**.

**Wick touch:** `Low ≤ EMA ≤ High`

**Close near EMA:** Close is within the selected percentage tolerance.

Only the **latest qualifying occurrence for each stock** is retained.
Results are ranked by occurrence date, newest first.

### Buying List

A stock is included only when its **latest RSI is below 35** and its
latest close is within **±2% of the EMA 255**.

The Buying List shows six fundamental and price indicators:

**PE | Sector PE | P/B | ROE | 52W Low | 52W High**

The **52W Range** column shows the stock's current position within its
52-week low to high range. Dividend yield is intentionally excluded.
"""
    )
