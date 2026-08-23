
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import (
    add_days_since_cross,
    calculate_indicators,
    convergence_table,
    fundamental_snapshot,
    latest_snapshot,
    load_universe,
    download_prices,
)


st.set_page_config(
    page_title="Nifty Market Terminal",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Styling. Helvetica first, then common system alternatives.
# -------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
    --bg: #080b10;
    --panel: #10151c;
    --panel-soft: #131a23;
    --border: #28323d;
    --text: #eef2f6;
    --muted: #9aa7b5;
    --accent: #f0a51a;
    --green: #27c78a;
    --red: #ef6b68;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: Helvetica, Arial, sans-serif;
}

html, body, [class*="css"] {
    font-family: Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 1440px;
    padding-top: 4.8rem !important;
    padding-bottom: 3rem;
}

[data-testid="stTopNav"] {
    z-index: 1000;
    min-height: 52px;
}

[data-testid="stSidebar"] {
    background: #0b0f14;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    font-family: Helvetica, Arial, sans-serif;
}

.terminal-topbar {
    background: linear-gradient(180deg, #111821 0%, #0d1218 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 16px 20px;
    margin: 0 0 18px 0;
    border-radius: 7px;
}

.terminal-kicker,
.page-kicker,
.section-kicker {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-size: 10px;
    font-weight: 700;
}

.terminal-brand {
    font-size: 28px;
    font-weight: 750;
    margin-top: 3px;
}

.terminal-sub {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
    margin-top: 5px;
}

.page-hero {
    padding: 8px 0 16px;
}

.page-title {
    font-size: 38px;
    font-weight: 750;
    line-height: 1.08;
    margin: 5px 0 8px;
}

.page-copy {
    color: var(--muted);
    max-width: 940px;
    font-size: 15px;
    line-height: 1.6;
}

.guide-box {
    background: #101720;
    border: 1px solid #354251;
    border-left: 3px solid var(--accent);
    border-radius: 7px;
    padding: 14px 16px;
    margin: 8px 0 18px;
}

.guide-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 5px;
}

.guide-copy {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

.guide-step {
    margin-top: 6px;
}

.info-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 16px;
    min-height: 125px;
}

.info-card h4 {
    margin: 0 0 8px;
    font-size: 15px;
}

.info-card p {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
    margin: 0;
}

.status-chip {
    display: inline-block;
    border: 1px solid #3b4653;
    background: #111820;
    color: var(--muted);
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 11px;
    margin: 0 4px 8px 0;
}

.status-chip.green {
    color: var(--green);
    border-color: #1f6a50;
}

.status-chip.orange {
    color: var(--accent);
    border-color: #735919;
}

.status-chip.red {
    color: var(--red);
    border-color: #69302e;
}

.section-title {
    font-size: 20px;
    font-weight: 750;
    margin: 22px 0 9px;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 6px;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 42px;
    border-radius: 5px;
    border: 1px solid #4a5665;
    background: #141b24;
    color: var(--text);
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent);
    background: #1a222c;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
}

.small-note {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.45;
}

hr {
    border-color: var(--border);
}

.product-pill {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    border: 1px solid #3a4653;
    background: #111821;
    color: #cbd5e1;
    font-size: 10px;
    letter-spacing: .7px;
    text-transform: uppercase;
    margin-right: 5px;
}
.product-pill.accent {
    color: #f0a51a;
    border-color: #735919;
}

</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------

def terminal_header(page_title: str, subtitle: str):
    universe_count = len(st.session_state.get("universe", []))
    scan_date = st.session_state.get("scan_date", "Not run")

    st.markdown(
        f"""
<div class="terminal-topbar">
  <div class="terminal-kicker">NIFTY MARKET TERMINAL</div>
  <div class="terminal-brand">{page_title}</div>
  <div class="terminal-sub">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if "snapshot" in st.session_state:
        status = '<span class="status-chip green">SCAN READY</span>'
    else:
        status = '<span class="status-chip orange">RUN A SCAN FIRST</span>'

    st.markdown(
        f"""
<div>
  {status}
  <span class="status-chip">Universe: {universe_count or "Not scanned"}</span>
  <span class="status-chip">Last scan: {scan_date}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def page_intro(title: str, copy: str):
    st.markdown(
        f"""
<div class="page-hero">
  <div class="page-kicker">Research guide</div>
  <div class="page-title">{title}</div>
  <div class="page-copy">{copy}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def guide(title: str, copy: str, steps=None):
    items = ""
    for i, step in enumerate(steps or [], start=1):
        items += f'<div class="guide-step"><b>{i}.</b> {step}</div>'

    st.markdown(
        f"""
<div class="guide-box">
  <div class="guide-title">{title}</div>
  <div class="guide-copy">{copy}{items}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def card(title: str, copy: str):
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-copy">{copy}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def require_scan():
    if "snapshot" not in st.session_state:
        st.info(
            "Run the market scan from **Scan Engine** first. "
            "All strategy pages reuse that cached scan."
        )
        st.stop()


def market_chart(
    frame: pd.DataFrame,
    symbol: str,
    overlays: list[str],
    rsi_col: str = "RSI14",
    days: int = 180,
    cross_columns: list[str] | None = None,
    rsi_lines: list[tuple[float, str]] | None = None,
):
    """Bloomberg-style price + strategy indicators + RSI chart."""
    chart = frame.sort_values("Date").tail(days).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.76, 0.24],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart["Date"],
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name=symbol,
            increasing_line_color="#26c281",
            increasing_fillcolor="#26c281",
            decreasing_line_color="#ef6461",
            decreasing_fillcolor="#ef6461",
        ),
        row=1,
        col=1,
    )

    colors = {
        "EMA9": "#6ea8fe",
        "EMA21": "#f0a51a",
        "SMA20": "#b084f5",
        "SMA50": "#14b8a6",
        "SMA200": "#ef4444",
        "EMA255": "#f59e0b",
    }

    for col in overlays:
        if col not in chart.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=chart["Date"],
                y=chart[col],
                mode="lines",
                name=col,
                line={"width": 1.8, "color": colors.get(col, "#cbd5e1")},
            ),
            row=1,
            col=1,
        )

    for cross_col in cross_columns or []:
        if cross_col not in chart.columns:
            continue
        marks = chart.loc[chart[cross_col].fillna(False)]
        if marks.empty:
            continue
        label = {
            "Cross9_21": "9/21 Bullish Cross",
            "Cross20_50": "20/50 Bullish Cross",
            "Cross50_200": "Golden Cross",
        }.get(cross_col, "Bullish Cross")
        fig.add_trace(
            go.Scatter(
                x=marks["Date"],
                y=marks["Close"],
                mode="markers",
                name=label,
                marker={
                    "symbol": "triangle-up",
                    "size": 9,
                    "color": "#26c281",
                    "line": {"color": "#080a0d", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    if rsi_col in chart.columns:
        fig.add_trace(
            go.Scatter(
                x=chart["Date"],
                y=chart[rsi_col],
                mode="lines",
                name=rsi_col,
                line={"width": 1.7, "color": "#8ab4ff"},
            ),
            row=2,
            col=1,
        )
        for level, label in (rsi_lines or [(30, "RSI 30"), (70, "RSI 70")]):
            fig.add_hline(
                y=level,
                row=2,
                col=1,
                line_dash="dot",
                line_color="#46515f",
                line_width=1,
                annotation_text=label,
                annotation_position="top left",
                annotation_font={"size": 9, "color": "#7f8b99"},
            )

    fig.update_layout(
        height=610,
        margin={"l": 8, "r": 8, "t": 40, "b": 10},
        paper_bgcolor="#080a0d",
        plot_bgcolor="#080a0d",
        font={"family": "Helvetica, Arial, sans-serif", "color": "#e9eef3"},
        legend={"orientation": "h", "y": 1.03, "x": 0, "font": {"size": 10}},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor="#1d232b", linecolor="#252c35", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="#1d232b", linecolor="#252c35", showline=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(
        gridcolor="#1d232b",
        linecolor="#252c35",
        showline=False,
        zeroline=False,
        range=[0, 100],
        title_text="RSI",
        title_font={"size": 10, "color": "#8c98a6"},
        row=2,
        col=1,
    )
    return fig


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------

def home_page():
    terminal_header(
        "Home",
        "A guided workspace for finding and researching Indian equity setups",
    )

    page_intro(
        "Find the signal. Check the context. Do your homework.",
        "Start with one market scan, explore a setup that matches your style, "
        "inspect the chart, and then move to the shortlist. You do not need to "
        "understand every indicator to use the app.",
    )

    guide(
        "Start here. Let's scan.",
        "The workflow is simple. The analysis underneath it is not.",
        [
            "Scan the market once. The result is reused across the app.",
            "Open Market Health, Momentum, Swing, or Pullback.",
            "Check the stock's chart before treating a signal as meaningful.",
            "Use Confluence and Shortlist to narrow your research list.",
        ],
    )

    snapshot = st.session_state.get("snapshot", pd.DataFrame())
    conv = st.session_state.get("convergence", pd.DataFrame())
    universe_count = len(st.session_state.get("universe", []))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks in universe", f"{universe_count:,}" if universe_count else "—")
    c2.metric("Stocks scanned", f"{len(snapshot):,}" if not snapshot.empty else "—")
    c3.metric(
        "Bullish long-term trend",
        f"{int(snapshot['BullRegime'].sum()):,}" if not snapshot.empty else "—",
    )
    c4.metric(
        "High-conviction candidates",
        f"{int((conv['ConvergenceScore'] >= 65).sum()):,}" if not conv.empty else "—",
    )

    st.markdown('<div class="section-title">What each section means</div>', unsafe_allow_html=True)

    cards = st.columns(4)
    sections = [
        (
            "Market Health",
            "The big-picture filter. Tells you whether the long-term trend is helping or fighting you.",
        ),
        (
            "Momentum",
            "Fast money, fast signals. Spots fresh short-term momentum using the 9/21 EMA.",
        ),
        (
            "Swing",
            "The calmer setup. Looks for multi-week trend alignment using the 20/50 averages.",
        ),
        (
            "Pullback",
            "Hunting the pullback. Finds oversold names sitting close to their long-term EMA 255.",
        ),
    ]

    for col, (title, text) in zip(cards, sections):
        with col:
            card(title, text)

    st.markdown('<div class="section-title">How to read a stock</div>', unsafe_allow_html=True)

    read = st.columns(3)
    with read[0]:
        card("Signal", "The rule that made the stock qualify.")
    with read[1]:
        card("Chart", "The price action that tells you whether the signal looks healthy or weak.")
    with read[2]:
        card("Fundamentals", "Valuation and business-quality fields to review after technical screening.")

    st.caption(
        "Research tool only. Signals are not investment recommendations and are not guarantees of future returns."
    )


def scan_page():
    terminal_header(
        "Scan Engine",
        "Run one shared scan and reuse it everywhere else",
    )

    page_intro(
        "Scan the market once.",
        "Choose your universe and history. The scanner downloads daily market data "
        "and calculates the indicators used by every strategy. You do not need to "
        "repeat the download for each page.",
    )

    guide(
        "Recommended setup",
        "For normal use, keep the defaults. A 4-year history gives enough context "
        "for the long moving averages while keeping the scan practical on free hosting.",
        [
            "Select Nifty Total Market for the broadest current universe.",
            "Keep 4 years of history.",
            "Click Run Market Scan and wait for the data-quality result.",
        ],
    )

    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])

    with c1:
        universe_name = st.selectbox(
            "Stock universe",
            ["NIFTY TOTAL MARKET", "NIFTY 500"],
            index=0,
            help="Nifty Total Market gives the broadest screen.",
        )

    with c2:
        history_years = st.selectbox(
            "Price history",
            [3, 4, 5],
            index=1,
        )

    with c3:
        batch_size = st.select_slider(
            "Download chunk",
            options=[50, 75, 100],
            value=75,
            help="Smaller chunks can be more resilient to public-data limits.",
        )

    with c4:
        run_scan = st.button(
            "RUN MARKET SCAN",
            type="primary",
            use_container_width=True,
        )

    if run_scan:
        try:
            with st.spinner("Loading the current stock universe..."):
                universe = load_universe(universe_name)

            progress = st.progress(0)
            status = st.empty()

            def update(batch, total, failures):
                progress.progress(batch / total)
                status.write(
                    f"Downloading market data: {batch}/{total} · "
                    f"unresolved symbols: {failures}"
                )

            prices, failures = download_prices(
                universe,
                years=history_years,
                batch_size=batch_size,
                progress_callback=update,
            )

            if prices.empty:
                st.error("No usable market data was returned.")
                st.stop()

            status.write("Calculating indicators...")
            indicators = calculate_indicators(prices)
            snapshot = latest_snapshot(indicators, universe)
            snapshot = add_days_since_cross(indicators, snapshot)
            convergence = convergence_table(snapshot)

            st.session_state["universe"] = universe
            st.session_state["prices"] = prices
            st.session_state["indicators"] = indicators
            st.session_state["snapshot"] = snapshot
            st.session_state["convergence"] = convergence
            st.session_state["failures"] = failures
            st.session_state["scan_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

            progress.empty()
            status.empty()

            st.success(
                f"Scan complete. {len(snapshot):,} stocks have usable daily history."
            )

        except Exception as exc:
            st.error(
                "The market scan could not be completed. "
                "Try again or reduce the download chunk."
            )
            with st.expander("Technical details"):
                st.exception(exc)

    if "snapshot" in st.session_state:
        snapshot = st.session_state["snapshot"]
        failures = st.session_state.get("failures", [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe", f"{len(st.session_state['universe']):,}")
        c2.metric("Usable price histories", f"{len(snapshot):,}")
        c3.metric("Bullish long-term trend", f"{int(snapshot['BullRegime'].sum()):,}")
        c4.metric("Oversold pullbacks", f"{int(snapshot['Pullback'].sum()):,}")

        st.markdown('<div class="section-title">Data quality</div>', unsafe_allow_html=True)

        q1, q2, q3 = st.columns(3)
        q1.metric(
            "Coverage",
            f"{len(snapshot) / max(len(st.session_state['universe']), 1) * 100:.1f}%",
        )
        q2.metric(
            "Latest market date",
            pd.to_datetime(snapshot["Date"]).max().strftime("%d %b %Y"),
        )
        q3.metric("Unresolved symbols", f"{len(failures):,}")

        if failures:
            st.warning(
                f"{len(failures)} symbols did not return usable history and were excluded."
            )

        st.download_button(
            "EXPORT MARKET SNAPSHOT",
            data=snapshot.to_csv(index=False).encode(),
            file_name="nifty_market_snapshot.csv",
            mime="text/csv",
        )

        guide(
            "You're ready",
            "Use the top menu to explore a strategy. The strategy pages reuse this exact scan."
        )


def strategy_page(strategy: str):
    require_scan()
    snapshot = st.session_state["snapshot"]
    indicators = st.session_state["indicators"]
    prices = st.session_state["prices"]

    titles = {
        "regime": (
            "Market Regime",
            "Long-term trend context using 50/200 SMA.",
        ),
        "momentum": (
            "9/21 EMA Momentum",
            "Short-term direction and fresh momentum turns.",
        ),
        "swing": (
            "20/50 Swing Structure",
            "Medium-term trend alignment.",
        ),
        "pullback": (
            "EMA 255 Pullback",
            "Oversold price near the long-term EMA.",
        ),
    }

    title, subtitle = titles[strategy]
    terminal_header(title, subtitle)

    explanations = {
        "regime": (
            "How to use Market Health",
            "This is the background environment. It asks whether the long-term structure is supportive.",
            [
                "Bullish structure means the 50-day average is above the 200-day average.",
                "Use this as context, not as a standalone buy signal.",
            ],
        ),
        "momentum": (
            "How to use Momentum",
            "This page finds short-term changes in direction using the 9 and 21 EMA.",
            [
                "Fresh Cross means the 9 EMA has recently crossed above the 21 EMA.",
                "Bullish Momentum means the 9 EMA is currently above the 21 EMA.",
                "Use the chart to judge whether the move is fresh or already extended.",
            ],
        ),
        "swing": (
            "How to use Swing",
            "This page looks for medium-term alignment between the 20-day and 50-day averages.",
            [
                "A bullish structure means the faster average is above the slower average.",
                "Check the long-term trend and chart before considering the setup.",
            ],
        ),
        "pullback": (
            "How to use Pullback",
            "This page finds stocks that are short-term oversold and close to EMA 255.",
            [
                "RSI below 35 means the stock is short-term oversold.",
                "Within ±2% of EMA 255 means price is close to the long-term reference.",
                "A pullback can keep falling. Use the chart to look for stabilisation.",
            ],
        ),
    }

    g_title, g_copy, g_steps = explanations[strategy]
    guide(g_title, g_copy, g_steps)

    if strategy == "regime":
        mask = snapshot["BullRegime"]
        table = snapshot.loc[
            mask,
            [
                "Symbol",
                "Company",
                "Close",
                "SMA50",
                "SMA200",
                "DaysSince50_200",
            ],
        ].copy()

        table["State"] = "BULLISH"

    elif strategy == "momentum":
        mode = st.radio(
            "View",
            ["Fresh Cross", "Bullish Momentum", "Bearish Momentum"],
            horizontal=True,
        )

        if mode == "Fresh Cross":
            mask = snapshot["MomentumFresh"]
        elif mode == "Bullish Momentum":
            mask = snapshot["BullMomentum"]
        else:
            mask = ~snapshot["BullMomentum"]

        table = snapshot.loc[
            mask,
            [
                "Symbol",
                "Company",
                "Close",
                "EMA9",
                "EMA21",
                "DaysSince9_21",
                "RSI14",
                "EMA255DistancePct",
            ],
        ].copy()

    elif strategy == "swing":
        mask = snapshot["BullSwing"]
        table = snapshot.loc[
            mask,
            [
                "Symbol",
                "Company",
                "Close",
                "SMA20",
                "SMA50",
                "DaysSince20_50",
                "RSI14",
                "EMA255DistancePct",
            ],
        ].copy()

    else:
        mask = snapshot["Pullback"]
        table = snapshot.loc[
            mask,
            [
                "Symbol",
                "Company",
                "Close",
                "RSI14",
                "EMA255DistancePct",
                "BullRegime",
                "BullSwing",
                "BullMomentum",
            ],
        ].copy()

    st.metric(
        "QUALIFYING STOCKS",
        f"{len(table):,}",
    )

    search = st.text_input(
        "Search",
        placeholder="Symbol or company",
    )

    if search:
        mask_text = (
            table["Symbol"].str.contains(search, case=False, na=False)
            | table["Company"].str.contains(search, case=False, na=False)
        )
        table = table.loc[mask_text]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    st.markdown('<div class="section-kicker">Chart console</div>', unsafe_allow_html=True)

    show_chart = st.toggle(
        "SHOW STRATEGY CHART",
        value=True,
        key=f"chart_toggle_{strategy}",
    )

    if show_chart and not table.empty:
        chart_col_1, chart_col_2 = st.columns([2, 1])

        with chart_col_1:
            chosen = st.selectbox(
                "Select stock",
                table["Symbol"].tolist(),
                key=f"chart_stock_{strategy}",
            )

        with chart_col_2:
            chart_days = st.selectbox(
                "Chart window",
                [90, 180, 252, 365],
                index=1,
                key=f"chart_days_{strategy}",
            )

        row = snapshot.loc[snapshot["Symbol"] == chosen].iloc[0]
        frame = indicators.loc[
            indicators["Yahoo Symbol"] == row["Yahoo Symbol"]
        ].copy()

        if strategy == "regime":
            overlays = ["SMA50", "SMA200"]
            crosses = ["Cross50_200"]
            rsi_levels = [(30, "RSI 30"), (70, "RSI 70")]
            chart_note = "50/200 regime + RSI context"
        elif strategy == "momentum":
            overlays = ["EMA9", "EMA21", "EMA255"]
            crosses = ["Cross9_21"]
            rsi_levels = [(50, "RSI 50"), (70, "RSI 70")]
            chart_note = "9/21 momentum + EMA 255 trend + RSI"
        elif strategy == "swing":
            overlays = ["SMA20", "SMA50", "EMA255"]
            crosses = ["Cross20_50"]
            rsi_levels = [(50, "RSI 50"), (70, "RSI 70")]
            chart_note = "20/50 swing structure + EMA 255 + RSI"
        else:
            overlays = ["EMA255"]
            crosses = []
            rsi_levels = [(35, "BUY ZONE 35"), (50, "RSI 50"), (70, "RSI 70")]
            chart_note = "EMA 255 pullback + RSI buy-zone"

        st.caption(chart_note)
        st.plotly_chart(
            market_chart(
                frame,
                chosen,
                overlays,
                rsi_col="RSI14",
                days=int(chart_days),
                cross_columns=crosses,
                rsi_lines=rsi_levels,
            ),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )


def convergence_page():
    require_scan()
    df = st.session_state["convergence"].copy()

    terminal_header(
        "Convergence Engine",
        "Trend and entry conditions without double-counting related signals.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "HIGH CONVICTION",
        f"{int((df['ConvergenceScore'] >= 65).sum()):,}",
    )
    c2.metric(
        "STRONG TREND",
        f"{int((df['TrendScore'] >= 60).sum()):,}",
    )
    c3.metric(
        "PULLBACK SETUPS",
        f"{int(df['Pullback'].sum()):,}",
    )

    min_score = st.slider(
        "Minimum convergence score",
        0,
        100,
        65,
        5,
    )

    output = df.loc[
        df["ConvergenceScore"] >= min_score,
        [
            "Symbol",
            "Company",
            "Close",
            "ConvergenceScore",
            "TrendScore",
            "EntryScore",
            "Setup",
            "BullRegime",
            "BullSwing",
            "BullMomentum",
            "Pullback",
            "RSI14",
            "EMA255DistancePct",
        ],
    ].copy()

    output.insert(
        0,
        "Rank",
        range(1, len(output) + 1),
    )

    st.dataframe(
        output,
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    st.download_button(
        "EXPORT CONVERGENCE CSV",
        data=output.to_csv(index=False).encode(),
        file_name="nifty_total_market_convergence.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-kicker">Convergence chart</div>', unsafe_allow_html=True)
    show_chart = st.toggle(
        "SHOW CONVERGENCE CHART",
        value=True,
        key="chart_toggle_convergence",
    )

    if show_chart and not output.empty:
        selected = st.selectbox(
            "Select stock",
            output["Symbol"].tolist(),
            key="chart_stock_convergence",
        )
        row = df.loc[df["Symbol"] == selected].iloc[0]
        frame = st.session_state["indicators"].loc[
            st.session_state["indicators"]["Yahoo Symbol"] == row["Yahoo Symbol"]
        ].copy()
        st.caption("All major moving averages + crossover markers + RSI 14")
        st.plotly_chart(
            market_chart(
                frame,
                selected,
                ["EMA9", "EMA21", "SMA20", "SMA50", "SMA200", "EMA255"],
                rsi_col="RSI14",
                days=252,
                cross_columns=["Cross9_21", "Cross20_50", "Cross50_200"],
                rsi_lines=[(30, "RSI 30"), (50, "RSI 50"), (70, "RSI 70")],
            ),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )


def buying_list_page():
    require_scan()
    df = st.session_state["convergence"].copy()

    terminal_header(
        "Final Buying List",
        "Research shortlist. Fundamentals are fetched only for finalists.",
    )

    shortlist = df.loc[
        (df["TrendScore"] >= 60)
        & (
            (df["EntryScore"] >= 15)
            | df["Pullback"]
            | df["MomentumFresh"]
        )
    ].copy()

    shortlist = shortlist.sort_values(
        ["ConvergenceScore", "TrendScore", "EntryScore"],
        ascending=False,
    ).head(25)

    if shortlist.empty:
        st.info(
            "No stocks currently meet the final shortlist rules."
        )
        return

    rows = []

    with st.spinner(
        f"Fetching fundamentals for {len(shortlist)} finalists..."
    ):
        for _, row in shortlist.iterrows():
            fund = fundamental_snapshot(row["Yahoo Symbol"])

            rows.append(
                {
                    "Symbol": row["Symbol"],
                    "Company": row["Company"],
                    "Setup": row["Setup"],
                    "Score": int(row["ConvergenceScore"]),
                    "Trend": int(row["TrendScore"]),
                    "Entry": int(row["EntryScore"]),
                    "RSI": round(float(row["RSI14"]), 2),
                    "EMA255 Dist %": round(
                        float(row["EMA255DistancePct"]),
                        2,
                    ),
                    "P/E": fund["PE"],
                    "P/B": fund["PB"],
                    "Margin %": fund["Profit Margin %"],
                    "Debt/Equity": fund["Debt/Equity"],
                    "EV/EBITDA": fund["EV/EBITDA"],
                    "Market Cap": fund["Market Cap"],
                    "Yahoo Symbol": row["Yahoo Symbol"],
                }
            )

    final = pd.DataFrame(rows)

    for column in [
        "P/E",
        "P/B",
        "Margin %",
        "Debt/Equity",
        "EV/EBITDA",
    ]:
        final[column] = pd.to_numeric(
            final[column],
            errors="coerce",
        ).round(2)

    final = final.reset_index(drop=True)

    st.metric(
        "FINAL RESEARCH CANDIDATES",
        f"{len(final):,}",
    )

    st.dataframe(
        final.drop(columns=["Yahoo Symbol"]),
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    st.download_button(
        "EXPORT FINAL BUYING LIST",
        data=final.drop(columns=["Yahoo Symbol"]).to_csv(index=False).encode(),
        file_name="nifty_total_market_buying_list.csv",
        mime="text/csv",
    )

    st.divider()

    show_chart = st.toggle(
        "SHOW CHART FOR BUYING LIST STOCK",
        value=False,
    )

    if show_chart:
        selected = st.selectbox(
            "Stock",
            final["Symbol"].tolist(),
        )

        row = final.loc[
            final["Symbol"] == selected
        ].iloc[0]

        frame = st.session_state["indicators"].loc[
            st.session_state["indicators"]["Yahoo Symbol"] == row["Yahoo Symbol"]
        ].copy()

        st.caption("All major moving averages + crossover markers + RSI 14")
        st.plotly_chart(
            market_chart(
                frame,
                selected,
                ["EMA9", "EMA21", "SMA20", "SMA50", "SMA200", "EMA255"],
                rsi_col="RSI14",
                days=252,
                cross_columns=["Cross9_21", "Cross20_50", "Cross50_200"],
                rsi_lines=[(30, "RSI 30"), (35, "BUY ZONE 35"), (50, "RSI 50"), (70, "RSI 70")],
            ),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )


# -------------------------------------------------------------------
# Navigation
# -------------------------------------------------------------------

# Named wrapper functions are used instead of lambdas so every page has
# a unique callable name and explicit URL path. This avoids duplicate
# page-path errors in Streamlit navigation.

def regime_page():
    strategy_page("regime")


def momentum_page():
    strategy_page("momentum")


def swing_page():
    strategy_page("swing")


def pullback_page():
    strategy_page("pullback")


pages = {
    "Start": [
        st.Page(
            home_page,
            title="Home",
            icon="🏠",
            url_path="home",
            default=True,
        ),
    ],
    "Scan": [
        st.Page(
            scan_page,
            title="Scan Engine",
            icon="🔄",
            url_path="scan-engine",
        ),
    ],
    "Explore": [
        st.Page(
            regime_page,
            title="Market Regime",
            icon="📐",
            url_path="market-regime",
        ),
        st.Page(
            momentum_page,
            title="9/21 Momentum",
            icon="📈",
            url_path="momentum-9-21",
        ),
        st.Page(
            swing_page,
            title="20/50 Swing",
            icon="📊",
            url_path="swing-20-50",
        ),
        st.Page(
            pullback_page,
            title="EMA 255 Pullback",
            icon="↔️",
            url_path="ema-255-pullback",
        ),
    ],
    "Decide": [
        st.Page(
            convergence_page,
            title="Confluence",
            icon="🎯",
            url_path="confluence",
        ),
        st.Page(
            buying_list_page,
            title="Final Buy List",
            icon="⭐",
            url_path="final-buy-list",
        ),
    ],
}

pg = st.navigation(
    pages,
    position="top",
)

with st.sidebar:
    st.markdown(
        """
<div class="section-kicker">Quick start</div>
<div class="small-note">
<b>1.</b> Run Scan Engine<br>
<b>2.</b> Explore a setup<br>
<b>3.</b> Check the chart<br>
<b>4.</b> Open Confluence<br>
<b>5.</b> Open Final Buy List
</div>
""",
        unsafe_allow_html=True,
    )

    if "snapshot" in st.session_state:
        st.success("SCAN READY")
    else:
        st.warning("RUN A MARKET SCAN")

    st.divider()

    st.caption(
        "Built for research. Public market data. "
        "No broker API key required."
    )

pg.run()
