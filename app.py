
import io
import time
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_nifty500_constituents():
    """
    Download and validate the current Nifty 500 constituent file.

    Two public URLs are attempted to improve resilience.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
    }

    errors = []

    for url in NIFTY500_URLS:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            normalized = {
                str(column).strip().lower(): column
                for column in df.columns
            }

            symbol_col = normalized.get("symbol")
            company_col = (
                normalized.get("company name")
                or normalized.get("company")
                or normalized.get("name")
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
                    f"Constituent file returned only {len(result)} rows."
                )

            result["Yahoo Symbol"] = result["Symbol"] + ".NS"
            return result

        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to download the Nifty 500 constituent list.\n"
        + "\n".join(errors)
    )


def _normalize_download(data, tickers):
    """Convert yfinance batch output into a single long-format DataFrame."""
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
            required = {"Close", "High", "Low"}

            if not frame.empty and required.issubset(frame.columns):
                frame = frame.reset_index()
                frame["Yahoo Symbol"] = ticker
                frames.append(frame)
    else:
        if len(tickers) == 1:
            frame = data.dropna(how="all").copy()
            required = {"Close", "High", "Low"}

            if not frame.empty and required.issubset(frame.columns):
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

    numeric_columns = [
        column
        for column in ["Open", "High", "Low", "Close", "Volume"]
        if column in result.columns
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result.dropna(
        subset=["Date", "High", "Low", "Close"]
    ).reset_index(drop=True)


@st.cache_data(
    ttl=60 * 60 * 4,
    show_spinner=False,
)
def download_price_batch(tickers_tuple, start_date, end_date):
    """
    Download one batch with retries.

    Cached by batch and date range, so reruns and other users can reuse
    recent downloads while the Streamlit server remains active.
    """
    tickers = list(tickers_tuple)
    last_error = None

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

            normalized = _normalize_download(data, tickers)

            if not normalized.empty:
                return normalized

        except Exception as exc:
            last_error = exc

        time.sleep(1.5 * (attempt + 1))

    # Return an empty DataFrame instead of killing the whole scan.
    return pd.DataFrame()


def download_all_prices(
    tickers,
    start_date,
    end_date,
    batch_size,
    progress_callback=None,
):
    """
    Download the Nifty 500 universe in manageable batches.

    If a batch partially fails, individual missing tickers are retried.
    """
    all_frames = []
    failures = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

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

        # Retry missing tickers individually. This prevents one bad ticker
        # from invalidating a whole batch.
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

        # Gentle pause between uncached batch requests.
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


# ---------------------------------------------------------
# Signal engine
# ---------------------------------------------------------

def classify_signal(
    frame,
    touch_mode,
    tolerance_pct,
):
    frame = frame.copy()
    frame["Previous Close"] = frame["Close"].shift(1)
    frame["Previous EMA"] = frame["EMA"].shift(1)

    if touch_mode == "Wick":
        touches = (
            frame["Low"].le(frame["EMA"])
            & frame["High"].ge(frame["EMA"])
        )
    else:
        distance = (
            (frame["Close"] - frame["EMA"]).abs()
            / frame["EMA"].abs()
        )
        touches = distance.le(tolerance_pct / 100)

    return touches & frame["EMA"].notna()


def scan_prices(
    universe,
    prices,
    ema_period,
    touch_mode,
    tolerance_pct,
):
    """
    Calculate the EMA independently for each stock and retain only
    its latest qualifying touch.
    """
    if prices.empty:
        return pd.DataFrame()

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

# Previous values are required to determine whether
# the stock touched the EMA from above or below.
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
        close_distance = abs(
            float(hit["Close"]) - float(hit["EMA"])
        ) / max(abs(float(hit["EMA"])), 1e-12)

        if close_distance <= 0.001:
            signal_type = "Close on EMA"
        elif pd.notna(previous_close) and pd.notna(previous_ema):
            if previous_close > previous_ema:
                signal_type = "Touched from Above"
            elif previous_close < previous_ema:
                signal_type = "Touched from Below"
            else:
                signal_type = "Touch"
        else:
            signal_type = "Touch"

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
        return pd.DataFrame()

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

    numeric_columns = [
        "Close",
        f"EMA {ema_period}",
        "Distance from EMA %",
    ]

    for column in numeric_columns:
        output[column] = output[column].round(2)

    output["Occurrence Date"] = output[
        "Occurrence Date"
    ].dt.strftime("%Y-%m-%d")

    return output


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------

def show_scan_progress(batch, total, failures):
    progress = batch / total
    st.session_state["_download_progress"] = progress
    st.session_state["_download_text"] = (
        f"Downloading batch {batch} of {total}"
        f" · unresolved tickers: {failures}"
    )


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("Nifty 500 EMA Scanner")
st.caption(
    "Free, API-key-free daily technical scanner for the Nifty 500."
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
        help=(
            "More history provides additional context but also increases "
            "the initial download size."
        ),
    )

    touch_label = st.radio(
        "Signal definition",
        options=[
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
        disabled=(touch_label == "Wick touches EMA"),
    )

    batch_size = st.select_slider(
        "Download batch size",
        options=[25, 50, 75, 100],
        value=50,
        help=(
            "Smaller batches are slower but can be more resilient if a "
            "provider request fails."
        ),
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
        st.session_state["_download_progress"] = 0.0
        st.session_state["_download_text"] = (
            "Preparing Nifty 500 universe..."
        )

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
            progress = batch / total
            progress_bar.progress(progress)
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
            f"Downloaded {prices['Yahoo Symbol'].nunique():,} stocks. "
            "Calculating EMA signals..."
        )

        touch_mode = (
            "Wick"
            if touch_label == "Wick touches EMA"
            else "Close"
        )

        with st.spinner(
            f"Calculating EMA {ema_period} and detecting signals..."
        ):
            signals = scan_prices(
                universe=universe,
                prices=prices,
                ema_period=int(ema_period),
                touch_mode=touch_mode,
                tolerance_pct=float(tolerance_pct),
            )

        st.session_state["signals"] = signals
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
        f"{len(meta['failures'])} ticker(s) could not be downloaded "
        "after retries. The remaining universe was scanned normally."
    )

with st.expander("Scan details"):
    st.write(
        {
            "Scan date": meta["scan_date"],
            "Constituents loaded": meta["constituents"],
            "Stocks with price data": meta["downloaded"],
            "Failed tickers": len(meta["failures"]),
            "EMA period": current_ema,
            "Signal definition": meta["touch_label"],
            "History downloaded": f"{meta['history_years']} years",
        }
    )

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Signals found",
    f"{len(signals):,}",
)

if signals.empty:
    metric_2.metric(
        "Most recent signal",
        "None",
    )
    metric_3.metric(
        "Signals in last 7 days",
        "0",
    )
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

st.subheader("Ranked EMA Signals")

filter_col_1, filter_col_2 = st.columns([2, 1])

with filter_col_1:
    search = st.text_input(
        "Search symbol or company",
        placeholder="RELIANCE, TCS, HDFC...",
    )

with filter_col_2:
    if signals.empty:
        max_age = 365
    else:
        max_age = max(
            1,
            int(
                (
                    pd.Timestamp.today().normalize()
                    - pd.to_datetime(
                        signals["Occurrence Date"]
                    ).min()
                ).days
            ),
        )

    max_days = st.number_input(
        "Only signals from last N days",
        min_value=1,
        max_value=max(3650, max_age),
        value=min(30, max(3650, max_age)),
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
    f"Showing {len(filtered):,} of {len(signals):,} detected signals."
)

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    height=600,
    column_config={
        "Rank": st.column_config.NumberColumn(
            "Rank",
            format="%d",
        ),
        "Close": st.column_config.NumberColumn(
            "Close",
            format="%.2f",
        ),
        f"EMA {current_ema}": st.column_config.NumberColumn(
            f"EMA {current_ema}",
            format="%.2f",
        ),
        "Distance from EMA %": st.column_config.NumberColumn(
            "Distance from EMA %",
            format="%.2f%%",
        ),
    },
)

csv = signals.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download nifty500_ema_signals.csv",
    data=csv,
    file_name="nifty500_ema_signals.csv",
    mime="text/csv",
    use_container_width=False,
)

if meta["failures"]:
    failed_csv = (
        pd.DataFrame(
            {
                "Yahoo Symbol": meta["failures"],
            }
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

**EMA:** Exponential Moving Average using **{current_ema}** daily closing prices.

**Wick touch:** `Low ≤ EMA ≤ High`

**Close near EMA:** `|Close - EMA| / EMA ≤ selected tolerance`

Only the **latest qualifying occurrence for each stock** is retained.
Results are ranked by occurrence date, newest first.

The app uses public market data through `yfinance`. Data availability can vary,
and a small number of tickers may occasionally fail or return incomplete history.
"""
    )
