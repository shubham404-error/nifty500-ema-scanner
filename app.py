
from __future__ import annotations

from datetime import datetime
import io
import json
import hashlib

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from google import genai
from google.genai import types
from plotly.subplots import make_subplots

# -------------------------------------------------------------------
# AI strategy constants. Defined before any shared strategy helpers use them.
# -------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash-lite"
AI_STRATEGY_PREFILTER_SCORE = 75
AI_DEFAULT_FINAL_BUY_CONVICTION = 78
AI_DEFAULT_FINAL_BUY_LIQUIDITY = True
AI_FUNDAMENTAL_FETCH_LIMIT = 25

from engine import (
    add_days_since_cross,
    calculate_indicators,
    convergence_table,
    fundamental_snapshot,
    investor_quality_gate,
    latest_snapshot,
    load_universe,
    download_prices,
)


st.set_page_config(
    page_title="Nifty Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
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

html, body {
    font-family: Helvetica, Arial, sans-serif;
}

.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stText"],
.stApp p,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp label {
    font-family: Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 1440px;
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0b0f14;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] button {
    font-family: Helvetica, Arial, sans-serif;
}

/* Never override Streamlit / Material Symbols fonts.
   Otherwise icon ligatures render as literal text such as keyboard_double_arrow. */
.material-symbols-rounded,
.material-symbols-outlined,
[class*="material-symbols"],
[data-testid="stSidebar"] [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
    font-feature-settings: "liga";
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


@media (max-width: 768px) {
    .block-container {
        padding-top: 5.25rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .terminal-topbar {
        margin-top: 0;
        padding: 14px 15px;
    }

    .terminal-brand {
        font-size: 22px;
    }

    .page-title {
        font-size: 30px;
    }

    .page-copy {
        font-size: 14px;
    }

    [data-testid="stSidebar"] {
        width: min(86vw, 360px);
    }
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



def ai_prefilter_note():
    st.caption(
        f"AI verification note. To keep strategy checks focused and efficient, "
        f"the AI uses a high-conviction candidate pool of {AI_STRATEGY_PREFILTER_SCORE}+ "
        f"before running its list verification. This does not change the visible "
        f"Confluence, Final Buy List, or Emerging strategy rules."
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
    """Price, volume and RSI chart. Missing optional columns are handled safely."""
    chart = frame.sort_values("Date").tail(days).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.66, 0.14, 0.20],
    )
    fig.add_trace(go.Candlestick(
        x=chart["Date"], open=chart["Open"], high=chart["High"],
        low=chart["Low"], close=chart["Close"], name=symbol,
        increasing_line_color="#26c281", increasing_fillcolor="#26c281",
        decreasing_line_color="#ef6461", decreasing_fillcolor="#ef6461",
    ), row=1, col=1)

    colors = {"EMA9":"#6ea8fe", "EMA21":"#f0a51a", "SMA20":"#b084f5",
              "SMA50":"#14b8a6", "SMA200":"#ef4444", "EMA255":"#f59e0b"}
    for col in overlays:
        if col in chart.columns:
            fig.add_trace(go.Scatter(
                x=chart["Date"], y=chart[col], mode="lines", name=col,
                line={"width":1.8, "color":colors.get(col,"#cbd5e1")},
            ), row=1, col=1)

    for cross_col in cross_columns or []:
        if cross_col not in chart.columns:
            continue
        marks=chart.loc[chart[cross_col].fillna(False)]
        if marks.empty:
            continue
        label={"Cross9_21":"9/21 Bullish Cross", "Cross20_50":"20/50 Bullish Cross", "Cross50_200":"Golden Cross"}.get(cross_col,"Bullish Cross")
        fig.add_trace(go.Scatter(
            x=marks["Date"], y=marks["Close"], mode="markers", name=label,
            marker={"symbol":"triangle-up","size":9,"color":"#26c281","line":{"color":"#080a0d","width":1}},
        ), row=1, col=1)

    if "Volume" in chart.columns:
        fig.add_trace(go.Bar(x=chart["Date"], y=chart["Volume"], name="Volume", marker_color="#64748b"), row=2, col=1)
    if "VolumeSMA20" in chart.columns:
        fig.add_trace(go.Scatter(x=chart["Date"], y=chart["VolumeSMA20"], mode="lines", name="20D Avg Vol", line={"width":1.3,"color":"#f0a51a"}), row=2, col=1)

    if rsi_col in chart.columns:
        fig.add_trace(go.Scatter(x=chart["Date"], y=chart[rsi_col], mode="lines", name=rsi_col, line={"width":1.7,"color":"#8ab4ff"}), row=3, col=1)
        for level,label in (rsi_lines or [(30,"RSI 30"),(70,"RSI 70")]):
            fig.add_hline(y=level,row=3,col=1,line_dash="dot",line_color="#46515f",line_width=1,
                          annotation_text=label,annotation_position="top left",annotation_font={"size":9,"color":"#7f8b99"})

    fig.update_layout(height=690, margin={"l":8,"r":8,"t":40,"b":10}, paper_bgcolor="#080a0d",
        plot_bgcolor="#080a0d", font={"family":"Helvetica, Arial, sans-serif","color":"#e9eef3"},
        legend={"orientation":"h","y":1.03,"x":0,"font":{"size":10}}, hovermode="x unified",
        xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False, xaxis3_rangeslider_visible=False)
    for row in (1,2,3):
        fig.update_yaxes(gridcolor="#1d232b",linecolor="#252c35",showline=False,zeroline=False,row=row,col=1)
    fig.update_yaxes(range=[0,100], title_text="RSI", title_font={"size":10,"color":"#8c98a6"}, row=3,col=1)
    return fig


# -------------------------------------------------------------------
# Pages# -------------------------------------------------------------------
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

    st.info("MARKET TIMING NOTICE. This terminal is designed for after-market-hours research using completed daily candles. Run the scan after the NSE market has closed and use the results to prepare your research or watchlist for the next session. It is not a live intraday scanner.")

    guide(
        "Start here. Let's scan.",
        "The workflow is simple. The analysis underneath it is not.",
        [
            "Scan the market once. The result is reused across the app.",
            "Explore Market Health, Momentum, Swing, Pullback, or Emerging Setups.",
            "Check the stock's chart and then use Confluence and Final Buy List to narrow your research list.",
            "Use the new Nifty AI Analyst to ask, in plain English, why a stock qualified, what it means, and what to watch next.",
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

    st.markdown('<div class="section-title">New. Nifty AI Analyst</div>', unsafe_allow_html=True)

    ai_col, ai_note_col = st.columns([2.2, 1])
    with ai_col:
        card(
            "Ask why a stock qualified",
            "Nifty AI Analyst is integrated into the terminal so you can select a stock and ask why it appeared in a strategy, what the setup means in plain English, what looks positive, and what to monitor next. It uses the terminal's current strategy and market context rather than acting as a separate stock-picking engine.",
        )
    with ai_note_col:
        st.info(
            f"AI verification uses a {AI_STRATEGY_PREFILTER_SCORE}+ high-conviction "
            "candidate pool for Confluence and buy-list checks. This improves efficiency "
            "without changing the visible strategy rules or rankings."
        )

    st.markdown('<div class="section-title">How to read a stock</div>', unsafe_allow_html=True)

    read = st.columns(3)
    with read[0]:
        card("Signal", "The rule that made the stock qualify.")
    with read[1]:
        card("Chart", "The price action that tells you whether the signal looks healthy or weak.")
    with read[2]:
        card("Fundamentals", "Valuation and business-quality fields to review after technical screening.")


    with st.expander("WHAT IS EMERGING SETUPS?", expanded=False):
        st.markdown(
            """
**Emerging Setups** is a separate discovery screen for newer stocks that do not yet
have enough trading history for the full long-term strategy suite.

It does **not** weaken the main scanner and it does **not** replace the 200-day SMA
or 255-day EMA requirements. Instead, it asks a different question:

> *"Does this newer stock already show promising short- and medium-term behaviour?"*

It uses indicators that can be assessed without SMA 200 or EMA 255, such as
**9/21 EMA momentum, 20/50 swing structure, RSI, relative strength, volume,
ATR and liquidity**.

Use it as an **early research/watchlist tool**, not as a replacement for the
Confluence or Final Buy List. A stock can graduate into the normal strategy suite
once enough history becomes available.
"""
        )

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
            ["NIFTY TOTAL MARKET", "NIFTY 500", "NIFTY 200", "NIFTY 50"],
            index=0,
            help="Nifty 50: focused large-cap. Nifty 200: large + mid-cap. Nifty 500: broad. Nifty Total Market: broadest.",
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
            scan_basis = snapshot[[c for c in ["Symbol", "Date", "Close"] if c in snapshot.columns]].copy()
            scan_id = "scan-" + hashlib.sha256(scan_basis.to_csv(index=False).encode()).hexdigest()[:16]
            _invalidate_strategy_state(scan_id)
            st.session_state["scan_id"] = scan_id

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
            "Your market scan is complete. If some stocks have insufficient history for the full strategy suite, review them in Emerging Setups below."
        )

        st.markdown(
            "[**→ OPEN EMERGING SETUPS**](/emerging-setups)",
            unsafe_allow_html=True,
        )


def strategy_page(strategy: str):
    require_scan()
    snapshot = st.session_state["snapshot"].copy()
    indicators = st.session_state["indicators"]

    titles = {
        "regime": ("Market Regime", "Long-term trend context using 50/200 SMA."),
        "momentum": ("9/21 EMA Momentum", "Short-term direction, fresh crosses and volume confirmation."),
        "swing": ("20/50 Swing Structure", "Medium-term alignment with relative-strength context."),
        "pullback": ("EMA 255 Pullback", "Oversold price near long-term support with quality context."),
    }
    title, subtitle = titles[strategy]
    terminal_header(title, subtitle)

    explanations = {
        "regime": ("WHAT DO 50 / 200 SMA MEAN?", "The 50-day average tracks the medium-term trend and the 200-day average is a long-term trend reference. When the 50 SMA is above the 200 SMA, the app treats long-term structure as bullish. This is context, not a standalone buy signal."),
        "momentum": ("WHAT DOES 9 / 21 EMA MOMENTUM MEAN?", "The 9 EMA reacts faster than the 21 EMA. A fresh 9-above-21 cross can signal strengthening short-term momentum. Established alignment means momentum is already positive, while volume confirmation adds evidence of participation."),
        "swing": ("WHAT DOES 20 / 50 SWING STRUCTURE MEAN?", "The 20-day and 50-day simple moving averages describe medium-term trend structure. A 20 SMA above the 50 SMA indicates bullish swing alignment. Relative strength and volume provide additional quality context."),
        "pullback": ("WHAT DOES EMA 255 PULLBACK MEAN?", "EMA 255 is used as a long-term trend reference. This setup looks for oversold price action near that level. RSI below 35 and proximity to EMA 255 create the trigger, but oversold does not automatically mean buy."),
    }
    expander_title, expander_text = explanations[strategy]
    with st.expander(expander_title, expanded=False):
        st.write(expander_text)

    filter_col, liquidity_col = st.columns([2,1])
    with liquidity_col:
        liquidity_filter = st.selectbox("Liquidity", ["All","₹1 Cr+","₹5 Cr+","₹25 Cr+"], index=0, key=f"liq_{strategy}")
    if strategy == "regime":
        table=snapshot.loc[snapshot["BullRegime"]].copy()
        columns=["Symbol","Company","Close","SMA50","SMA200","DaysSince50_200","RS3MPct","AvgTradedValue20","LiquidityBucket"]
        table["State"]="BULLISH"
        columns.append("State")
    elif strategy == "momentum":
        with filter_col:
            mode=st.radio("View",["Fresh Cross","Bullish Momentum","Volume Confirmed","20D Breakout"],horizontal=True)
        masks={"Fresh Cross":snapshot["MomentumFresh"],"Bullish Momentum":snapshot["BullMomentum"],"Volume Confirmed":snapshot["VolumeConfirmedMomentum"],"20D Breakout":snapshot["Breakout20"]}
        table=snapshot.loc[masks[mode]].copy()
        columns=["Symbol","Company","Close","EMA9","EMA21","DaysSince9_21","RSI14","RS3MPct","VolumeRatio","AvgTradedValue20","LiquidityBucket","ATRPercent"]
    elif strategy == "swing":
        table=snapshot.loc[snapshot["BullSwing"]].copy()
        columns=["Symbol","Company","Close","SMA20","SMA50","DaysSince20_50","RSI14","RS3MPct","RS6MPct","VolumeRatio","AvgTradedValue20","ATRPercent"]
    else:
        table=snapshot.loc[snapshot["Pullback"]].copy()
        columns=["Symbol","Company","Close","RSI14","EMA255DistancePct","BullRegime","BullSwing","RS3MPct","VolumeRatio","AvgTradedValue20","LiquidityBucket","ATRPercent"]

    thresholds={"All":0,"₹1 Cr+":1_00_00_000,"₹5 Cr+":5_00_00_000,"₹25 Cr+":25_00_00_000}
    table=table.loc[table["AvgTradedValue20"].fillna(0)>=thresholds[liquidity_filter], columns].copy()
    st.metric("QUALIFYING STOCKS", f"{len(table):,}")

    search=st.text_input("Search",placeholder="Symbol or company",key=f"search_{strategy}")
    if search:
        table=table.loc[table["Symbol"].str.contains(search,case=False,na=False)|table["Company"].str.contains(search,case=False,na=False)]
    st.dataframe(table,use_container_width=True,hide_index=True,height=520,column_config={
        "RS3MPct":st.column_config.NumberColumn("RS 3M %ile",format="%.0f"),
        "RS6MPct":st.column_config.NumberColumn("RS 6M %ile",format="%.0f"),
        "VolumeRatio":st.column_config.NumberColumn("Volume x",format="%.2f"),
        "AvgTradedValue20":st.column_config.NumberColumn("20D Traded Value",format="₹ %.0f"),
        "ATRPercent":st.column_config.NumberColumn("ATR %",format="%.2f%%"),
    })

    st.markdown('<div class="section-kicker">Chart console</div>',unsafe_allow_html=True)
    show_chart=st.toggle("SHOW STRATEGY CHART",value=True,key=f"chart_toggle_{strategy}")
    if show_chart and not table.empty:
        c1,c2=st.columns([2,1])
        with c1: chosen=st.selectbox("Select stock",table["Symbol"].tolist(),key=f"chart_stock_{strategy}")
        with c2: chart_days=st.selectbox("Chart window",[90,180,252,365],index=1,key=f"chart_days_{strategy}")
        row=snapshot.loc[snapshot["Symbol"]==chosen].iloc[0]
        frame=indicators.loc[indicators["Yahoo Symbol"]==row["Yahoo Symbol"]].copy()
        if strategy=="regime": overlays=["SMA50","SMA200"]; crosses=["Cross50_200"]; levels=[(30,"RSI 30"),(70,"RSI 70")]
        elif strategy=="momentum": overlays=["EMA9","EMA21","EMA255"]; crosses=["Cross9_21"]; levels=[(50,"RSI 50"),(70,"RSI 70")]
        elif strategy=="swing": overlays=["SMA20","SMA50","EMA255"]; crosses=["Cross20_50"]; levels=[(50,"RSI 50"),(70,"RSI 70")]
        else: overlays=["EMA255"]; crosses=[]; levels=[(35,"BUY ZONE 35"),(50,"RSI 50"),(70,"RSI 70")]
        st.caption("Price structure, 20-day average volume and RSI are shown together.")
        st.plotly_chart(market_chart(frame,chosen,overlays,rsi_col="RSI14",days=int(chart_days),cross_columns=crosses,rsi_lines=levels),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})


def _ai_register_strategy_output(strategy_name, frame, metadata=None):
    """Central, exact strategy registry for the AI layer."""
    if "ai_strategy_registry" not in st.session_state:
        st.session_state["ai_strategy_registry"] = {}

    st.session_state["ai_strategy_registry"][strategy_name] = {
        "frame": frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(),
        "metadata": metadata or {},
    }


def convergence_page():
    require_scan()
    df=st.session_state["convergence"].copy()
    terminal_header("Confluence","Setup-aware ranking. Trend, pullback, fresh momentum and breakout paths are evaluated separately.")

    with st.expander("WHAT DO THESE COLUMNS MEAN?", expanded=False):
        st.markdown("**Setup** is the qualifying strategy path. **Convergence Score** measures that path strength. **RS 3M %ile** compares strength within the scanned universe. **Volume x** compares current participation with normal volume. **Liquidity** uses 20-day average traded value. **ATR %** is volatility.")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("ACTIVE SETUPS",f"{int((df['Setup']!='No active setup').sum()):,}")
    c2.metric("SCORE 80+",f"{int((df['ConvergenceScore']>=80).sum()):,}")
    c3.metric("VOLUME BREAKOUTS",f"{int(df['Breakout20'].sum()):,}")
    c4.metric("LIQUID",f"{int(df['LiquidityEligible'].sum()):,}")

    f1,f2=st.columns(2)
    with f1: min_score=st.slider("Minimum setup score",0,100,60,5)
    with f2: setup_filter=st.selectbox("Setup type",["All"]+sorted([x for x in df['Setup'].dropna().unique() if x!='No active setup']))
    output=df.loc[(df["ConvergenceScore"]>=min_score)&(df["Setup"]!="No active setup")].copy()
    if setup_filter!="All": output=output.loc[output["Setup"]==setup_filter]
    st.session_state["ai_confluence_output"] = output.copy()
    _ai_register_strategy_output(
        "confluence",
        output,
        {
            "minimum_score": float(min_score),
            "setup_filter": setup_filter,
        },
    )
    cols=["Symbol","Company","Setup","Close","ConvergenceScore","TrendContinuationScore","PullbackScore","FreshMomentumScore","BreakoutScore","RS3MPct","VolumeRatio","LiquidityBucket","ATRPercent","RSI14","EMA255DistancePct"]
    output=output[cols].copy().reset_index(drop=True)
    output.insert(0,"Rank",range(1,len(output)+1))
    st.dataframe(output,use_container_width=True,hide_index=True,height=560)
    st.download_button("EXPORT CONFLUENCE CSV",data=output.to_csv(index=False).encode(),file_name="nifty_total_market_confluence.csv",mime="text/csv")

    st.markdown('<div class="section-kicker">Confluence chart</div>',unsafe_allow_html=True)
    if st.toggle("SHOW CONVERGENCE CHART",value=True,key="chart_toggle_convergence") and not output.empty:
        selected=st.selectbox("Select stock",output["Symbol"].tolist(),key="chart_stock_convergence")
        row=df.loc[df["Symbol"]==selected].iloc[0]
        frame=st.session_state["indicators"].loc[st.session_state["indicators"]["Yahoo Symbol"]==row["Yahoo Symbol"]].copy()
        st.plotly_chart(market_chart(frame,selected,["EMA9","EMA21","SMA20","SMA50","SMA200","EMA255"],rsi_col="RSI14",days=252,cross_columns=["Cross9_21","Cross20_50","Cross50_200"],rsi_lines=[(30,"RSI 30"),(35,"BUY ZONE 35"),(50,"RSI 50"),(70,"RSI 70")]),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})


def _current_scan_id():
    """Stable identity for the current scan. Prevents stale strategy outputs."""
    existing = st.session_state.get("scan_id")
    if existing:
        return str(existing)
    snapshot = st.session_state.get("snapshot")
    if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
        basis = snapshot[[c for c in ["Symbol", "Date", "Close"] if c in snapshot.columns]].copy()
        digest = hashlib.sha256(basis.to_csv(index=False).encode()).hexdigest()[:16]
        return f"scan-{digest}"
    return "no-scan"


def _invalidate_strategy_state(scan_id):
    """Drop all strategy/AI artifacts that belong to an older scan."""
    st.session_state["ai_strategy_registry"] = {}
    for key in [
        "ai_confluence_output", "ai_final_buy_output", "ai_emerging_working",
        "ai_emerging_buy_output", "ai_final_buy_cache", "ai_emerging_cache",
        "buy_fundamentals_cache", "emerging_fundamentals_cache",
    ]:
        st.session_state.pop(key, None)
    st.session_state["strategy_scan_id"] = scan_id


def _strategy_cache_valid(cache):
    return isinstance(cache, dict) and cache.get("scan_id") == _current_scan_id()


def _fundamental_for_scan(namespace, yahoo_symbol):
    scan_id = _current_scan_id()
    cache = st.session_state.setdefault(namespace, {})
    key = f"{scan_id}|{yahoo_symbol}"
    if key not in cache:
        try:
            result = fundamental_snapshot(yahoo_symbol)
            cache[key] = result if isinstance(result, dict) else {}
        except Exception:
            cache[key] = {}
    return cache[key]


def build_ai_confluence_pool(convergence=None, min_score=AI_STRATEGY_PREFILTER_SCORE):
    """Canonical AI Confluence pool using an explicit score threshold and active setups only."""
    if convergence is None:
        convergence = st.session_state.get("convergence")
    if not isinstance(convergence, pd.DataFrame) or convergence.empty:
        return pd.DataFrame()
    score = pd.to_numeric(convergence.get("ConvergenceScore"), errors="coerce")
    setup = convergence.get("Setup", pd.Series("No active setup", index=convergence.index))
    return convergence.loc[
        (score >= float(min_score))
        & setup.astype(str).ne("No active setup")
    ].copy().reset_index(drop=True)


def build_final_buy_list(
    convergence=None,
    min_score=AI_DEFAULT_FINAL_BUY_CONVICTION,
    use_liquidity=AI_DEFAULT_FINAL_BUY_LIQUIDITY,
    prefilter_score=None,
):
    """Shared Final Buy List engine used by both the page and AI verification."""
    if convergence is None:
        convergence = st.session_state.get("convergence")
    if not isinstance(convergence, pd.DataFrame) or convergence.empty:
        return pd.DataFrame()

    source = convergence.copy()
    if prefilter_score is not None:
        source = build_ai_confluence_pool(source, min_score=prefilter_score)

    technical = investor_quality_gate(source)
    if use_liquidity and not technical.empty:
        technical = technical.loc[technical["LiquidityEligible"].fillna(False)].copy()
    if technical.empty:
        return pd.DataFrame()

    finalists = technical.head(AI_FUNDAMENTAL_FETCH_LIMIT).copy()
    rows = []
    for _, row in finalists.iterrows():
        fund = _fundamental_for_scan("buy_fundamentals_cache", row.get("Yahoo Symbol"))
        market_cap = pd.to_numeric(fund.get("Market Cap"), errors="coerce")
        revenue_growth = pd.to_numeric(fund.get("Revenue Growth %"), errors="coerce")
        margin = pd.to_numeric(fund.get("Profit Margin %"), errors="coerce")
        debt_equity = pd.to_numeric(fund.get("Debt/Equity"), errors="coerce")
        pe = pd.to_numeric(fund.get("PE"), errors="coerce")
        fundamental_score = (
            (2 if pd.notna(market_cap) and market_cap >= 5_00_00_00_000 else 0)
            + (2 if pd.notna(revenue_growth) and revenue_growth > 0 else 0)
            + (2 if pd.notna(margin) and margin > 0 else 0)
            + (2 if pd.notna(debt_equity) and 0 <= debt_equity <= 2.5 else 0)
            + (2 if pd.notna(pe) and 0 < pe <= 60 else 0)
        )
        conviction = float(row["InvestorTechnicalScore"]) * 0.90 + fundamental_score
        rows.append({
            "Symbol": row.get("Symbol"), "Company": row.get("Company"), "Setup": row.get("Setup"),
            "Investor Conviction": round(conviction, 1),
            "Technical Quality": round(float(row["InvestorTechnicalScore"]), 1),
            "Confluence": int(pd.to_numeric(row.get("ConvergenceScore"), errors="coerce")),
            "RS 3M %ile": round(float(row["RS3MPct"]), 1) if pd.notna(row.get("RS3MPct")) else None,
            "RS 6M %ile": round(float(row["RS6MPct"]), 1) if pd.notna(row.get("RS6MPct")) else None,
            "Volume x": round(float(row["VolumeRatio"]), 2) if pd.notna(row.get("VolumeRatio")) else None,
            "Liquidity": row.get("LiquidityBucket"),
            "RSI": round(float(row["RSI14"]), 2) if pd.notna(row.get("RSI14")) else None,
            "ATR %": round(float(row["ATRPercent"]), 2) if pd.notna(row.get("ATRPercent")) else None,
            "Gap %": round(float(row["GapPct"]), 2) if pd.notna(row.get("GapPct")) else None,
            "P/E": fund.get("PE"), "Revenue Growth %": fund.get("Revenue Growth %"),
            "Net Profit Margin %": fund.get("Profit Margin %"), "Debt/Equity": fund.get("Debt/Equity"),
            "EV/EBITDA": fund.get("EV/EBITDA"), "Market Cap": fund.get("Market Cap"),
            "Yahoo Symbol": row.get("Yahoo Symbol"),
        })
    final = pd.DataFrame(rows)
    if final.empty:
        return final
    for col in ["Investor Conviction", "Technical Quality"]:
        final[col] = pd.to_numeric(final[col], errors="coerce")
    final = final.loc[final["Investor Conviction"] >= float(min_score)].sort_values(
        ["Investor Conviction", "Technical Quality", "RS 3M %ile"], ascending=False, na_position="last"
    ).reset_index(drop=True)
    if not final.empty:
        final.insert(0, "Rank", range(1, len(final) + 1))
    return final


def _build_emerging_scored(snapshot=None):
    """Shared Emerging scoring engine. No UI and no navigation dependency."""
    if snapshot is None:
        snapshot = st.session_state.get("snapshot")
    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty or "DataQualityStatus" not in snapshot.columns:
        return pd.DataFrame()
    working = snapshot.loc[snapshot["DataQualityStatus"].astype(str).str.strip().eq("Insufficient history")].copy()
    if working.empty:
        return working

    def num(frame, col):
        return pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(float("nan"), index=frame.index)
    score = pd.Series(0.0, index=working.index)
    for col, pts in [("BullMomentum",15),("BullSwing",15),("MomentumFresh",10),("VolumeConfirmedMomentum",10),("Breakout20",10)]:
        if col in working.columns:
            score += working[col].fillna(False).astype(bool).astype(float) * pts
    score += num(working,"RS3MPct").clip(0,100).fillna(0) * 0.15
    rsi = num(working,"RSI14")
    score += pd.Series(0.0,index=working.index).mask(rsi.between(50,70),5.0).mask(rsi.between(45,49.999999),3.0).mask(rsi.between(40,44.999999),1.0).mask(rsi.between(70.000001,75),3.0).fillna(0)
    working["Technical Score"] = score.clip(0,80).round(1)
    for col in ["Fundamental Score","Fundamental Coverage"]: working[col]=0.0
    working["Fundamental Coverage"] = 0
    for col in ["Revenue Growth %","Profit Margin %","Debt/Equity","P/E","EV/EBITDA","Market Cap"]: working[col]=float("nan")
    candidates = working.sort_values(["Technical Score","RS3MPct","VolumeRatio"],ascending=False,na_position="last").head(AI_FUNDAMENTAL_FETCH_LIMIT)
    for idx,row in candidates.iterrows():
        fund = _fundamental_for_scan("emerging_fundamentals_cache", row.get("Yahoo Symbol"))
        vals={
            "Revenue Growth %":pd.to_numeric(fund.get("Revenue Growth %"),errors="coerce"),
            "Profit Margin %":pd.to_numeric(fund.get("Profit Margin %"),errors="coerce"),
            "Debt/Equity":pd.to_numeric(fund.get("Debt/Equity"),errors="coerce"),
            "P/E":pd.to_numeric(fund.get("PE"),errors="coerce"),
            "EV/EBITDA":pd.to_numeric(fund.get("EV/EBITDA"),errors="coerce"),
            "Market Cap":pd.to_numeric(fund.get("Market Cap"),errors="coerce"),
        }
        rg,mg,de,pe,ev=vals["Revenue Growth %"],vals["Profit Margin %"],vals["Debt/Equity"],vals["P/E"],vals["EV/EBITDA"]
        fscore=(5 if pd.notna(rg) and rg>=15 else 3.5 if pd.notna(rg) and rg>0 else 1.5 if pd.notna(rg) and rg>-10 else 0)+(5 if pd.notna(mg) and mg>=15 else 3.5 if pd.notna(mg) and mg>0 else 1 if pd.notna(mg) and mg>-5 else 0)+(4 if pd.notna(de) and 0<=de<=1 else 3 if pd.notna(de) and de<=2 else 1.5 if pd.notna(de) and de<=3 else 0)+(3 if pd.notna(pe) and 0<pe<=30 else 2 if pd.notna(pe) and pe<=50 else 1 if pd.notna(pe) and pe<=75 else 0)+(3 if pd.notna(ev) and 0<ev<=20 else 2 if pd.notna(ev) and ev<=30 else 1 if pd.notna(ev) and ev<=50 else 0)
        for k,v in vals.items(): working.at[idx,k]=v
        working.at[idx,"Fundamental Score"]=round(min(fscore,20.0),1)
        working.at[idx,"Fundamental Coverage"]=sum(pd.notna(vals[k]) for k in ["Revenue Growth %","Profit Margin %","Debt/Equity","P/E","EV/EBITDA"])
    working["Emerging Score"]=(working["Technical Score"]+working["Fundamental Score"]).clip(0,100).round(1)
    def label(r):
        m=bool(r.get("BullMomentum",False)); sw=bool(r.get("BullSwing",False)); fr=bool(r.get("MomentumFresh",False)); br=bool(r.get("Breakout20",False)); vo=bool(r.get("VolumeConfirmedMomentum",False))
        if br and vo and m:return "Trend + Volume Breakout"
        if m and sw and fr:return "Fresh Momentum + Swing"
        if m and sw:return "Momentum + Swing"
        if br:return "Breakout Watch"
        if m:return "Momentum Watch"
        if sw:return "Swing Watch"
        return "Early Watch"
    working["Setup"]=working.apply(label,axis=1)
    return working


def build_emerging_buy_list(working):
    if not isinstance(working,pd.DataFrame) or working.empty:return pd.DataFrame()
    num=lambda c: pd.to_numeric(working[c],errors="coerce") if c in working.columns else pd.Series(float("nan"),index=working.index)
    buy=working.loc[num("Emerging Score")>=75].copy()
    buy=buy.loc[pd.to_numeric(buy["Technical Score"],errors="coerce")>=58].copy()
    buy=buy.loc[(pd.to_numeric(buy["RS3MPct"],errors="coerce") if "RS3MPct" in buy.columns else pd.Series(float("nan"),index=buy.index))>=70].copy()
    momentum=buy.get("BullMomentum",pd.Series(False,index=buy.index)).fillna(False).astype(bool)
    swing=buy.get("BullSwing",pd.Series(False,index=buy.index)).fillna(False).astype(bool)
    buy=buy.loc[momentum|swing].copy()
    buy=buy.loc[pd.to_numeric(buy["Fundamental Coverage"],errors="coerce")>=3].copy()
    buy=buy.loc[pd.to_numeric(buy["Fundamental Score"],errors="coerce")>=9].copy()
    traded=pd.to_numeric(buy["AvgTradedValue20"],errors="coerce") if "AvgTradedValue20" in buy.columns else pd.Series(0,index=buy.index)
    buy=buy.loc[traded.fillna(0)>=1_00_00_000].copy()
    return buy.sort_values(["Emerging Score","Fundamental Score","Technical Score","RS3MPct"],ascending=False,na_position="last").reset_index(drop=True)


def _ensure_ai_strategy_outputs():
    """Self-sufficient AI strategy build for the current scan only."""
    scan_id=_current_scan_id()
    registry=st.session_state.get("ai_strategy_registry",{})
    if not isinstance(registry,dict) or st.session_state.get("strategy_scan_id")!=scan_id:
        _invalidate_strategy_state(scan_id)
        registry={}
    convergence=st.session_state.get("convergence")
    confluence=build_ai_confluence_pool(convergence)
    final=build_final_buy_list(convergence,AI_DEFAULT_FINAL_BUY_CONVICTION,AI_DEFAULT_FINAL_BUY_LIQUIDITY,prefilter_score=AI_STRATEGY_PREFILTER_SCORE)
    emerging=_build_emerging_scored(st.session_state.get("snapshot"))
    emerging_buy=build_emerging_buy_list(emerging)
    _ai_register_strategy_output("confluence",confluence,{"minimum_score":AI_STRATEGY_PREFILTER_SCORE,"canonical_ai_pool":True,"scan_id":scan_id})
    _ai_register_strategy_output("final_buy_list",final,{"minimum_investor_conviction":AI_DEFAULT_FINAL_BUY_CONVICTION,"use_liquidity_filter":AI_DEFAULT_FINAL_BUY_LIQUIDITY,"prefilter_score":AI_STRATEGY_PREFILTER_SCORE,"canonical_ai_pool":True,"scan_id":scan_id})
    _ai_register_strategy_output("emerging_setups",emerging,{"scan_id":scan_id})
    _ai_register_strategy_output("emerging_buy_list",emerging_buy,{"scan_id":scan_id})
    st.session_state["ai_confluence_output"]=confluence.copy(); st.session_state["ai_final_buy_output"]=final.copy(); st.session_state["ai_emerging_working"]=emerging.copy(); st.session_state["ai_emerging_buy_output"]=emerging_buy.copy()


def buying_list_page():
    require_scan()

    df = st.session_state["convergence"].copy()

    terminal_header(
        "Final Buy List",
        "Strict retail-investor quality filter applied after Confluence. "
        "This is a research shortlist, not an automated buy recommendation.",
    )

    with st.expander("WHAT DO THE FINAL BUY LIST COLUMNS MEAN?", expanded=False):
        st.markdown("**Investor Conviction** combines final technical and fundamental quality checks. **Technical Quality** is the stricter second-stage screen. **Confluence** is the qualifying setup score. **RS**, **Volume x**, **Liquidity** and **ATR %** provide trend, participation, tradability and volatility context. **P/E, Revenue Growth, Net Profit Margin, Debt/Equity, EV/EBITDA and Market Cap** are research context and sanity checks, not automatic buy signals.")

    guide(
        "How this list is different",
        "Confluence finds valid technical setups. This page is deliberately stricter. "
        "It keeps only candidates with strong relative strength, setup-specific confirmation, "
        "controlled risk and usable liquidity.",
        [
            "Start with the strict technical quality gate.",
            "Keep the recommended liquidity filter on for practical retail execution.",
            "Fundamentals are fetched only for the strongest technical finalists.",
            "Use the final list for research and chart review, not blind execution.",
        ],
    )

    min_score = st.slider(
        "Minimum Investor Conviction Score",
        65,
        95,
        78,
        1,
        key="buy_conviction_score",
    )

    use_liquidity = st.toggle(
        "Use liquidity filter (recommended)",
        value=True,
        key="buy_liquidity_filter",
        help="Optional. Uses the existing 20-day average traded-value threshold.",
    )

    with st.spinner("Building the Final Buy List from the current scan..."):
        final = build_final_buy_list(
            df,
            min_score=min_score,
            use_liquidity=use_liquidity,
            prefilter_score=None,
        )

    if final.empty:
        st.info(
            "No technically and fundamentally strong candidates cleared the current Final Buy List rules."
        )
        return

    technical = investor_quality_gate(df)
    if use_liquidity and not technical.empty:
        technical = technical.loc[technical["LiquidityEligible"].fillna(False)].copy()
    fundamental_fetch_limit = AI_FUNDAMENTAL_FETCH_LIMIT

    st.session_state["ai_final_buy_output"] = final.copy()
    _ai_register_strategy_output(
        "final_buy_list",
        final,
        {
            "minimum_investor_conviction": float(min_score),
            "use_liquidity_filter": bool(use_liquidity),
            "scan_id": _current_scan_id(),
            "canonical_ai_pool": False,
        },
    )

    for col in [
        "P/E",
        "Revenue Growth %",
        "Net Profit Margin %",
        "Debt/Equity",
        "EV/EBITDA",
    ]:
        final[col] = pd.to_numeric(
            final[col],
            errors="coerce",
        ).round(2)

    st.metric(
        "FINAL RESEARCH CANDIDATES",
        f"{len(final):,}",
    )

    if len(technical) > fundamental_fetch_limit:
        st.caption(
            f"{len(technical)} technical candidates passed the quality gate. "
            f"Fundamentals were fetched only for the top {fundamental_fetch_limit} "
            "technical finalists to limit public-data requests."
        )

    display_cols = [
        "Rank",
        "Symbol",
        "Company",
        "Setup",
        "Investor Conviction",
        "Technical Quality",
        "Confluence",
        "RS 3M %ile",
        "RS 6M %ile",
        "Volume x",
        "Liquidity",
        "RSI",
        "ATR %",
        "P/E",
        "Revenue Growth %",
        "Net Profit Margin %",
        "Debt/Equity",
        "EV/EBITDA",
        "Market Cap",
    ]

    display_cols = [
        col for col in display_cols
        if col in final.columns
    ]

    st.dataframe(
        final[display_cols],
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    export = final.drop(
        columns=["Yahoo Symbol"],
        errors="ignore",
    )

    st.download_button(
        "EXPORT FINAL BUY LIST",
        data=export.to_csv(index=False).encode(),
        file_name="nifty_total_market_buying_list.csv",
        mime="text/csv",
    )

    st.divider()

    if st.toggle(
        "SHOW CHART FOR BUY LIST STOCK",
        value=False,
        key="buy_chart_toggle",
    ):
        selected = st.selectbox(
            "Stock",
            final["Symbol"].tolist(),
            key="buy_chart_stock",
        )
        row = final.loc[
            final["Symbol"] == selected
        ].iloc[0]

        frame = st.session_state["indicators"].loc[
            st.session_state["indicators"]["Yahoo Symbol"]
            == row["Yahoo Symbol"]
        ].copy()

        st.plotly_chart(
            market_chart(
                frame,
                selected,
                [
                    "EMA9",
                    "EMA21",
                    "SMA20",
                    "SMA50",
                    "SMA200",
                    "EMA255",
                ],
                rsi_col="RSI14",
                days=252,
                cross_columns=[
                    "Cross9_21",
                    "Cross20_50",
                    "Cross50_200",
                ],
                rsi_lines=[
                    (30, "RSI 30"),
                    (35, "BUY ZONE 35"),
                    (50, "RSI 50"),
                    (70, "RSI 70"),
                ],
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )




def emerging_setups_page():
    """
    Emerging Setups is an app.py-only discovery screen.

    It deliberately uses the existing market snapshot and DataQualityStatus
    classification. No market-data download, bars calculation, merge, or
    engine.py modification is performed here.
    """
    require_scan()

    snapshot = st.session_state.get("snapshot")
    indicators = st.session_state.get("indicators")
    if snapshot is None or snapshot.empty:
        st.warning("RUN A MARKET SCAN")
        return

    df = snapshot.copy()
    if "DataQualityStatus" not in df.columns:
        st.error(
            "The current market snapshot does not contain DataQualityStatus. "
            "Please run the current Scan Engine."
        )
        return

    # Source of truth for this page. Do not reconstruct history eligibility.
    working = df.loc[
        df["DataQualityStatus"].astype(str).str.strip().eq("Insufficient history")
    ].copy()

    terminal_header(
        "Emerging Setups",
        "Early-stage research for newer stocks. A separate scoring lane for names that do not yet qualify for the full long-term strategy suite.",
    )

    with st.expander("WHAT IS EMERGING SETUPS?", expanded=False):
        st.markdown(
            """
**Emerging Setups** is the early-discovery lane for stocks that the existing
scanner has already classified as **Insufficient history**.

It intentionally does **not** require **SMA 200** or **EMA 255**. Those
long-term indicators belong to the normal strategy suite once sufficient
history exists.

The page combines the shorter-history technical evidence already present in
the market snapshot with public fundamentals when available. It is designed
to find promising names early, not to bypass the stricter normal scanner.
"""
        )

    with st.expander("HOW TO USE THIS PAGE AS A RETAIL INVESTOR", expanded=False):
        st.markdown(
            """
1. **Start with Emerging Score**, then inspect the individual components.
2. **Prefer trend agreement**. 9 EMA above 21 EMA and 20 SMA above 50 SMA are stronger together than either alone.
3. **Look for participation**. Volume confirmation and a 20-day breakout make price strength more credible.
4. **Check relative strength**. A high 3-month percentile means the stock is outperforming much of the scanned universe.
5. **Use fundamentals as a second filter**, not as a replacement for the chart. Growth, profitability, leverage and valuation can improve or weaken the case.
6. **Use the Emerging Buy List only as a research shortlist**. Confirm the chart, liquidity, valuation, business quality and your own risk before investing.
"""
        )

    with st.expander("WHAT DO THESE INDICATORS AND SCORES MEAN?", expanded=False):
        st.markdown(
            """
### Emerging Score. 100 points

| Component | Points | What it rewards |
|---|---:|---|
| Bull Momentum | 15 | 9 EMA above 21 EMA |
| Bull Swing | 15 | 20 SMA above 50 SMA |
| Fresh Momentum | 10 | Recent 9/21 bullish momentum change |
| Volume Confirmation | 10 | Momentum supported by stronger participation |
| 20D Breakout | 10 | Recent breakout signal from the existing scanner |
| Relative Strength | 15 | 3-month percentile versus the scanned universe |
| RSI Context | 5 | Constructive momentum without blindly rewarding extreme RSI |
| Fundamentals | 20 | Growth, profitability, leverage and valuation when available |

**Technical score = 80. Fundamental score = 20.** Missing public fundamentals are
neutral, but the page reports fundamental coverage so a high score cannot be
mistaken for fully researched financial quality.

The score is a **ranking and discovery tool**, not a probability of return and
not a guaranteed buy signal.
"""
        )

    if working.empty:
        st.info(
            "There are currently no stocks classified as Insufficient history in the latest market snapshot."
        )
        return

    with st.spinner("Scoring emerging candidates from the current scan..."):
        working = _build_emerging_scored(snapshot)

    if working.empty:
        st.info("There are currently no emerging candidates after scoring the latest market snapshot.")
        return

    st.session_state["ai_emerging_working"] = working.copy()
    _ai_register_strategy_output(
        "emerging_setups",
        working,
        {"scan_id": _current_scan_id()},
    )

    def numeric_series(frame, column):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(float("nan"), index=frame.index)

    # -----------------------------
    # Controls.
    # -----------------------------
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        min_score = st.slider(
            "Minimum Emerging Score", 0, 100, 60, 5,
            key="emerging_min_score_v2",
        )
    with c2:
        liquidity_filter = st.selectbox(
            "Liquidity", ["All", "₹1 Cr+", "₹5 Cr+", "₹25 Cr+"],
            key="emerging_liquidity_v2",
        )
    with c3:
        bullish_only = st.toggle(
            "Prefer bullish structure", value=True,
            key="emerging_bullish_structure_v2",
        )

    liquidity_thresholds = {
        "All": 0,
        "₹1 Cr+": 1_00_00_000,
        "₹5 Cr+": 5_00_00_000,
        "₹25 Cr+": 25_00_00_000,
    }

    filtered = working.loc[working["Emerging Score"] >= min_score].copy()
    if bullish_only:
        bullish_mask = (
            filtered.get("BullMomentum", pd.Series(False, index=filtered.index)).fillna(False).astype(bool)
            | filtered.get("BullSwing", pd.Series(False, index=filtered.index)).fillna(False).astype(bool)
        )
        filtered = filtered.loc[bullish_mask].copy()

    traded_value = numeric_series(filtered, "AvgTradedValue20").fillna(0)
    filtered = filtered.loc[
        traded_value >= liquidity_thresholds[liquidity_filter]
    ].copy()

    filtered = filtered.sort_values(
        ["Emerging Score", "Technical Score", "RS3MPct", "Fundamental Score"],
        ascending=False,
        na_position="last",
    )

    # -----------------------------
    # Summary cards.
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("INSUFFICIENT-HISTORY STOCKS", f"{len(working):,}")
    c2.metric("PASSING SCREEN", f"{len(filtered):,}")
    c3.metric(
        "EMERGING SCORE 75+",
        f"{int((working['Emerging Score'] >= 75).sum()):,}",
    )
    c4.metric(
        "BULLISH STRUCTURE",
        f"{int(((working.get('BullMomentum', False).fillna(False).astype(bool)) & (working.get('BullSwing', False).fillna(False).astype(bool))).sum()):,}",
    )

    st.caption(
        "Source of history eligibility: DataQualityStatus from the existing Nifty Market Snapshot. "
        "SMA 200 and EMA 255 are intentionally not required."
    )

    # -----------------------------
    # Compact discovery table.
    # -----------------------------
    st.markdown('<div class="section-title">Emerging Research Candidates</div>', unsafe_allow_html=True)
    display_cols = [
        "Symbol", "Company", "Close", "Emerging Score", "Technical Score",
        "Fundamental Score", "Setup", "RS3MPct", "RSI14", "VolumeRatio",
        "LiquidityBucket", "Fundamental Coverage",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "Close": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "Emerging Score": st.column_config.NumberColumn("Score", format="%.1f"),
            "Technical Score": st.column_config.NumberColumn("Technical", format="%.1f"),
            "Fundamental Score": st.column_config.NumberColumn("Fundamentals", format="%.1f"),
            "RS3MPct": st.column_config.NumberColumn("RS 3M %ile", format="%.0f"),
            "RSI14": st.column_config.NumberColumn("RSI", format="%.1f"),
            "VolumeRatio": st.column_config.NumberColumn("Volume x", format="%.2f"),
            "Fundamental Coverage": st.column_config.NumberColumn("Fund. Coverage", format="%d/5"),
        },
    )

    st.download_button(
        "EXPORT EMERGING SETUPS CSV",
        data=filtered.to_csv(index=False).encode(),
        file_name="emerging_setups.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Actionable research shortlist.
    # -----------------------------
    st.markdown('<div class="section-title">Emerging Buy List</div>', unsafe_allow_html=True)
    st.caption(
        "Strict research shortlist. This is deliberately harder to pass than the discovery table and is not an automated investment recommendation."
    )

    buy = build_emerging_buy_list(working)
    st.session_state["ai_emerging_buy_output"] = buy.copy()
    _ai_register_strategy_output(
        "emerging_buy_list",
        buy,
        {"scan_id": _current_scan_id()},
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("EMERGING BUY CANDIDATES", f"{len(buy):,}")
    b2.metric("MIN SCORE", "75")
    b3.metric("MIN RS", "70th %ile")

    if buy.empty:
        st.info(
            "No emerging stock currently clears the strict Emerging Buy List. "
            "That is intentional. The screen does not force a buy candidate."
        )
    else:
        buy_cols = [
            "Symbol", "Company", "Close", "Emerging Score", "Setup",
            "Technical Score", "Fundamental Score", "Fundamental Coverage",
            "RS3MPct", "VolumeRatio", "LiquidityBucket", "Revenue Growth %",
            "Profit Margin %", "Debt/Equity", "P/E", "EV/EBITDA",
        ]
        buy_cols = [c for c in buy_cols if c in buy.columns]
        st.dataframe(
            buy[buy_cols],
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "Close": st.column_config.NumberColumn("Price", format="₹ %.2f"),
                "Emerging Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "Technical Score": st.column_config.NumberColumn("Technical", format="%.1f"),
                "Fundamental Score": st.column_config.NumberColumn("Fundamentals", format="%.1f"),
                "Fundamental Coverage": st.column_config.NumberColumn("Fund. Coverage", format="%d/5"),
                "RS3MPct": st.column_config.NumberColumn("RS 3M %ile", format="%.0f"),
                "VolumeRatio": st.column_config.NumberColumn("Volume x", format="%.2f"),
                "Revenue Growth %": st.column_config.NumberColumn("Revenue Growth", format="%.1f%%"),
                "Profit Margin %": st.column_config.NumberColumn("Margin", format="%.1f%%"),
                "Debt/Equity": st.column_config.NumberColumn("D/E", format="%.2f"),
                "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
                "EV/EBITDA": st.column_config.NumberColumn("EV/EBITDA", format="%.1f"),
            },
        )
        st.download_button(
            "EXPORT EMERGING BUY LIST",
            data=buy.to_csv(index=False).encode(),
            file_name="emerging_buy_list.csv",
            mime="text/csv",
        )

    # -----------------------------
    # Chart console. Uses already downloaded indicator history only.
    # -----------------------------
    st.markdown('<div class="section-title">Emerging Chart Console</div>', unsafe_allow_html=True)
    if indicators is not None and not filtered.empty:
        show_chart = st.toggle(
            "SHOW MOVING-AVERAGE CHART",
            value=True,
            key="emerging_chart_toggle_v2",
        )
        if show_chart:
            chart_symbols = filtered["Symbol"].tolist()
            selected = st.selectbox(
                "Select stock",
                chart_symbols,
                key="emerging_chart_stock_v2",
            )
            row = filtered.loc[filtered["Symbol"] == selected].iloc[0]
            frame = indicators.loc[
                indicators["Yahoo Symbol"] == row["Yahoo Symbol"]
            ].copy()
            if not frame.empty:
                overlays = [c for c in ["EMA9", "EMA21", "SMA20", "SMA50"] if c in frame.columns]
                st.caption("Price, 9/21 EMA, 20/50 SMA, 20-day average volume and RSI. Long-term SMA 200 and EMA 255 are intentionally excluded from this screen.")
                st.plotly_chart(
                    market_chart(
                        frame,
                        selected,
                        overlays,
                        rsi_col="RSI14",
                        days=252,
                        cross_columns=[c for c in ["Cross9_21", "Cross20_50"] if c in frame.columns],
                        rsi_lines=[(40, "RSI 40"), (50, "RSI 50"), (70, "RSI 70")],
                    ),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                )
            else:
                st.info("No indicator history is available for the selected stock.")

    with st.expander("IMPORTANT: WHAT THIS PAGE DOES NOT MEAN", expanded=False):
        st.markdown(
            """
A high Emerging Score is **not** equivalent to a normal Confluence score.

The Emerging Buy List is intentionally strict, but it remains a **research
shortlist** because these stocks do not yet have the long-term history required
by the main strategy suite. As history builds, the stock can graduate into the
normal scanner and be evaluated by the full Confluence and Final Buy List logic.
"""
        )


# -------------------------------------------------------------------
# Nifty AI Analyst. Read-only layer over the existing scan output.
# This section does not modify engine.py, scores, or scanner state.
# -------------------------------------------------------------------

# Constants are defined near the imports because shared strategy helpers use them earlier in the file.


def _ai_chart_png(frame, symbol):
    if frame is None or frame.empty:
        return None

    plot = frame.tail(180).copy()
    x = pd.to_datetime(plot["Date"], errors="coerce") if "Date" in plot.columns else plot.index

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    if "Close" in plot.columns:
        ax1.plot(x, pd.to_numeric(plot["Close"], errors="coerce"), label="Close", linewidth=1.6)

    for col in ["EMA9", "EMA21", "SMA20", "SMA50", "SMA200", "EMA255"]:
        if col in plot.columns:
            ax1.plot(x, pd.to_numeric(plot[col], errors="coerce"), label=col, linewidth=1.0)

    ax1.set_title(f"{symbol} . Existing terminal indicator history")
    ax1.legend(loc="upper left", ncol=3, fontsize=8)
    ax1.grid(alpha=0.2)

    if "RSI14" in plot.columns:
        ax2.plot(x, pd.to_numeric(plot["RSI14"], errors="coerce"), label="RSI14", linewidth=1.2)
        for level in [30, 50, 70]:
            ax2.axhline(level, linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.2)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _ai_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _ai_find_row(frame, symbol):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "Symbol" not in frame.columns:
        return None
    rows = frame.loc[frame["Symbol"].astype(str) == str(symbol)]
    return rows.iloc[0] if not rows.empty else None


def _ai_registered_strategy_frame(strategy_name, legacy_key=None):
    registry = st.session_state.get("ai_strategy_registry", {})
    entry = registry.get(strategy_name, {}) if isinstance(registry, dict) else {}
    metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
    frame = entry.get("frame") if isinstance(entry, dict) else None
    if metadata.get("scan_id") == _current_scan_id() and isinstance(frame, pd.DataFrame):
        return frame, metadata
    return None, {}


def _ai_strategy_status(symbol, snapshot_row):
    """Exact membership from the current scan. Never inferred by Gemini or navigation order."""
    _ensure_ai_strategy_outputs()
    status = {
        "confluence": {"member": False, "verified": False, "reason": None, "details": {}},
        "final_buy_list": {"member": False, "verified": False, "reason": None, "details": {}},
        "emerging_setups": {"member": False, "verified": False, "reason": None, "details": {}},
        "emerging_buy_list": {"member": False, "verified": False, "reason": None, "details": {}},
    }

    # Confluence. Prefer the exact filtered output currently shown in the app.
    confluence_frame, confluence_meta = _ai_registered_strategy_frame(
        "confluence",
        "ai_confluence_output",
    )
    row = _ai_find_row(confluence_frame, symbol)
    if row is not None:
        setup = str(row.get("Setup", "Active setup"))
        score = pd.to_numeric(row.get("ConvergenceScore"), errors="coerce")
        status["confluence"] = {
            "member": True,
            "verified": True,
            "reason": (
                f"Currently in the Confluence output. Setup: {setup}. "
                f"Confluence score: {float(score):.1f}."
            ),
            "details": {
                "setup": setup,
                "confluence_score": _ai_json_value(score),
                "minimum_score_used": confluence_meta.get("minimum_score"),
                "trend_score": _ai_json_value(row.get("TrendContinuationScore")),
                "pullback_score": _ai_json_value(row.get("PullbackScore")),
                "fresh_momentum_score": _ai_json_value(row.get("FreshMomentumScore")),
                "breakout_score": _ai_json_value(row.get("BreakoutScore")),
            },
        }

    # Final Buy List. This is the exact table, including rank and live threshold.
    final_frame, final_meta = _ai_registered_strategy_frame(
        "final_buy_list",
        "ai_final_buy_output",
    )
    row = _ai_find_row(final_frame, symbol)
    if row is not None:
        setup = str(row.get("Setup", "Current setup"))
        conviction = _ai_json_value(row.get("Investor Conviction"))
        technical_quality = _ai_json_value(row.get("Technical Quality"))
        confluence_score = _ai_json_value(row.get("Confluence"))
        rs3 = _ai_json_value(row.get("RS 3M %ile"))
        volume = _ai_json_value(row.get("Volume x"))
        liquidity = _ai_json_value(row.get("Liquidity"))

        reason_parts = [
            "Currently in the Final Buy List",
            f"Rank {row.get('Rank')}" if pd.notna(row.get("Rank")) else None,
            f"Setup: {setup}",
            f"Investor Conviction: {conviction}" if conviction is not None else None,
            f"Technical Quality: {technical_quality}" if technical_quality is not None else None,
            f"Confluence: {confluence_score}" if confluence_score is not None else None,
        ]
        reason = ". ".join(str(x) for x in reason_parts if x) + "."

        status["final_buy_list"] = {
            "member": True,
            "verified": True,
            "reason": reason,
            "details": {
                "rank": _ai_json_value(row.get("Rank")),
                "setup": setup,
                "investor_conviction": conviction,
                "technical_quality": technical_quality,
                "confluence": confluence_score,
                "rs_3m_percentile": rs3,
                "rs_6m_percentile": _ai_json_value(row.get("RS 6M %ile")),
                "volume_multiple": volume,
                "liquidity": liquidity,
                "minimum_investor_conviction_used": final_meta.get(
                    "minimum_investor_conviction"
                ),
                "liquidity_filter_used": final_meta.get("use_liquidity_filter"),
            },
        }

    # Emerging Setups.
    emerging_frame, _ = _ai_registered_strategy_frame(
        "emerging_setups",
        "ai_emerging_working",
    )
    row = _ai_find_row(emerging_frame, symbol)
    if row is not None:
        status["emerging_setups"] = {
            "member": True,
            "verified": True,
            "reason": (
                f"Currently in Emerging Setups. Setup: {row.get('Setup')}. "
                f"Emerging Score: {row.get('Emerging Score')}."
            ),
            "details": {
                "setup": _ai_json_value(row.get("Setup")),
                "emerging_score": _ai_json_value(row.get("Emerging Score")),
                "technical_score": _ai_json_value(row.get("Technical Score")),
                "fundamental_score": _ai_json_value(row.get("Fundamental Score")),
                "rs_3m_percentile": _ai_json_value(row.get("RS3MPct")),
                "fundamental_coverage": _ai_json_value(row.get("Fundamental Coverage")),
            },
        }

    # Emerging Buy List.
    emerging_buy_frame, _ = _ai_registered_strategy_frame(
        "emerging_buy_list",
        "ai_emerging_buy_output",
    )
    row = _ai_find_row(emerging_buy_frame, symbol)
    if row is not None:
        status["emerging_buy_list"] = {
            "member": True,
            "verified": True,
            "reason": (
                f"Currently in the Emerging Buy List. "
                f"Emerging Score: {row.get('Emerging Score')}."
            ),
            "details": {
                "setup": _ai_json_value(row.get("Setup")),
                "emerging_score": _ai_json_value(row.get("Emerging Score")),
                "technical_score": _ai_json_value(row.get("Technical Score")),
                "fundamental_score": _ai_json_value(row.get("Fundamental Score")),
                "fundamental_coverage": _ai_json_value(row.get("Fundamental Coverage")),
            },
        }

    return status

def _ai_stock_context(symbol):
    snapshot = st.session_state.get("snapshot")
    indicators = st.session_state.get("indicators")

    if snapshot is None or snapshot.empty:
        return None, None, None

    rows = snapshot.loc[snapshot["Symbol"].astype(str) == str(symbol)]
    if rows.empty:
        return None, None, None

    row = rows.iloc[0]
    row_data = {str(col): _ai_json_value(row[col]) for col in snapshot.columns}

    yahoo_symbol = row_data.get("Yahoo Symbol")
    history = None
    if indicators is not None and not indicators.empty and "Yahoo Symbol" in indicators.columns:
        history = indicators.loc[
            indicators["Yahoo Symbol"].astype(str) == str(yahoo_symbol)
        ].copy()

    strategy_status = _ai_strategy_status(symbol, row)

    context = {
        "symbol": str(symbol),
        "yahoo_symbol": str(yahoo_symbol) if yahoo_symbol else None,
        "data_date": str(row_data.get("Date", "")),
        "current_terminal_snapshot": row_data,
        "strategy_status": strategy_status,
        "rules": {
            "scores_are_engine_output": True,
            "ai_must_not_recalculate_or_change_scores": True,
            "missing_data_must_be_called_missing": True,
            "strategy_membership_must_come_from_strategy_status": True,
        },
    }

    return context, history, _ai_chart_png(history, symbol)


def _ai_compact_context(context):
    """Fixed-size factual packet. Interpretation is left to Gemini."""
    snapshot = context.get("current_terminal_snapshot", {})
    preferred = [
        "Symbol", "Company", "Date", "Close", "Price", "Last Price",
        "RSI14", "EMA9", "EMA21", "SMA20", "SMA50", "SMA200", "EMA255",
        "RS3MPct", "RS6MPct", "VolumeRatio", "Volume", "AvgVolume",
        "AvgTradedValue20", "LiquidityBucket", "ATRPercent",
        "DataQualityStatus", "Fundamental Coverage",
        "BullMomentum", "BullSwing", "Breakout20", "VolumeConfirmedMomentum",
    ]
    lower_lookup = {str(k).lower(): k for k in snapshot.keys()}
    compact = {}
    for wanted in preferred:
        actual = lower_lookup.get(wanted.lower())
        if actual is not None:
            compact[str(actual)] = snapshot[actual]

    return {
        "symbol": context.get("symbol"),
        "data_date": context.get("data_date"),
        "strategy_status": context.get("strategy_status", {}),
        "technical_snapshot": compact,
    }


def _ai_question_needs_fundamentals(question):
    q = str(question).lower()
    terms = [
        "fundamental", "valuation", "value", "expensive", "cheap", "overvalued",
        "undervalued", "company", "business", "revenue", "sales", "earnings",
        "profit", "margin", "debt", "balance sheet", "cash flow", "roe", "roa",
        "financial", "long term", "1 year", "2 year", "3 year", "investment",
        "hold", "quality company", "growth"
    ]
    return any(term in q for term in terms)


def _ai_question_is_clearly_fundamental_only(question):
    q = str(question).lower()
    fundamental = _ai_question_needs_fundamentals(q)
    technical_terms = [
        "entry", "buy now", "technical", "chart", "setup", "rsi", "ema", "sma",
        "support", "resistance", "breakout", "trend", "price action", "pullback",
        "momentum", "extended", "volume", "stop loss"
    ]
    return fundamental and not any(term in q for term in technical_terms)


def _ai_fetch_fundamental_packet(context):
    """Fetch and cache a richer Yahoo Finance packet for AI research questions."""
    ticker = context.get("yahoo_symbol")
    if not ticker:
        return {"available": False, "reason": "Yahoo Finance symbol is unavailable."}

    scan_id = _current_scan_id() or "no-scan"
    cache = st.session_state.setdefault("ai_fundamental_cache", {})
    cache_key = f"{scan_id}::{ticker}"
    if cache_key in cache:
        return cache[cache_key]

    fields = {
        "company": ["longName", "shortName", "sector", "industry"],
        "valuation": ["marketCap", "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda"],
        "growth": ["revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth"],
        "profitability": ["profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets"],
        "balance_sheet": ["totalCash", "totalDebt", "debtToEquity", "currentRatio", "quickRatio"],
        "cash_flow": ["operatingCashflow", "freeCashflow"],
        "share_statistics": ["sharesOutstanding", "floatShares", "heldPercentInsiders", "heldPercentInstitutions"],
    }
    pct_fields = {"revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth", "profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets", "heldPercentInsiders", "heldPercentInstitutions"}
    try:
        info = yf.Ticker(ticker).info or {}
        packet = {"available": bool(info), "source": "Yahoo Finance", "ticker": ticker}
        for section, names in fields.items():
            values = {}
            for name in names:
                value = info.get(name)
                if value is not None:
                    if name in pct_fields and isinstance(value, (int, float)):
                        value = value * 100
                    values[name] = _ai_json_value(value)
            packet[section] = values
        if not info:
            packet["reason"] = "Yahoo Finance returned no usable fundamental snapshot."
    except Exception as exc:
        packet = {"available": False, "source": "Yahoo Finance", "ticker": ticker, "reason": str(exc)[:180]}

    cache[cache_key] = packet
    return packet


def _ai_question_needs_chart(question):
    """Default to chart context. Skip only for clearly fundamental-only questions."""
    return not _ai_question_is_clearly_fundamental_only(question)


def _gemini_reply(question, context, chart_png, history):
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Streamlit Secrets.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )

    conversation = [
        {"role": item["role"], "content": item["content"]}
        for item in history[-6:]
    ]

    fundamental_packet = None
    if _ai_question_needs_fundamentals(question):
        fundamental_packet = _ai_fetch_fundamental_packet(context)

    instruction = """
You are Nifty AI, a practical research copilot inside a retail-focused Indian market terminal.

You are given factual terminal data. Treat strategy membership, rank, setup names, and numerical
scores as facts from the application. Do not invent or alter those facts.

Beyond those facts, reason independently. You may discuss the relationship between momentum,
entry timing, valuation, business quality, growth, profitability, leverage, and risk. Do not force
your answer to mechanically repeat the scanner's conclusion.

Important distinction:
- Strategy status answers whether the stock currently qualifies under a terminal strategy.
- Entry timing answers whether the current price appears attractive, extended, or risky.
These can point in different directions without contradiction. Explain that distinction naturally
when it matters instead of treating either one as automatically decisive.

Use the strongest verified strategy membership first when the user asks about qualification:
Final Buy List, Emerging Buy List, Confluence, then Emerging Setups.
Do not guess membership. If a field is missing, unavailable, or unsupported by the supplied data,
say so plainly.

Fundamental data, when supplied, is a Yahoo Finance snapshot. Use it as context, not as a
substitute for audited filings. Avoid pretending a single ratio proves that a company is good or bad.
If fundamentals and technicals tell different stories, explain the disagreement.

Answer for a reasonably informed retail investor:
- Be direct and conversational.
- Prefer plain English.
- Explain jargon briefly when useful.
- Do not dump every available metric.
- Use short sections and bullets only when they improve clarity.
- Do not invent news, events, targets, support levels, or financial figures not present in the data.
- Do not guarantee returns or give personalised financial advice.

Do not blindly apply generic heuristics such as 'RSI above 70 means do not buy'. Interpret the
indicator in the context of the supplied strategy, chart, and other evidence.
"""

    payload = {
        "question": str(question),
        "research_packet": _ai_compact_context(context),
        "fundamentals": fundamental_packet,
        "recent_conversation": conversation,
    }

    parts = [
        types.Part.from_text(text=instruction),
        types.Part.from_text(
            text="Research packet:\n" + json.dumps(payload, default=str, ensure_ascii=False)
        ),
    ]

    if chart_png and _ai_question_needs_chart(question):
        parts.append(types.Part.from_bytes(data=chart_png, mime_type="image/png"))

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
    except Exception as exc:
        raise RuntimeError(
            "Gemini did not respond within 30 seconds or the request was rejected. "
            f"Details: {exc}"
        ) from exc

    answer = getattr(response, "text", None)
    if not answer:
        raise RuntimeError("Gemini returned no usable text. Try a shorter question.")
    return answer

def nifty_ai_page():
    ai_prefilter_note()
    require_scan()

    snapshot = st.session_state.get("snapshot")
    if snapshot is None or snapshot.empty:
        st.warning("RUN A MARKET SCAN")
        return

    if "Symbol" not in snapshot.columns:
        st.error("The current scan does not contain a Symbol column.")
        return

    st.title("🤖 Nifty AI Analyst")
    st.caption(
        "Read-only AI interpretation of the current terminal scan. "
        "The existing engine and scores remain the source of truth."
    )

    symbols = sorted(snapshot["Symbol"].dropna().astype(str).unique().tolist())
    selected = st.selectbox("Select stock", symbols, key="nifty_ai_selected_stock")

    context, frame, chart_png = _ai_stock_context(selected)
    if context is None:
        st.error("Unable to build AI context for the selected stock.")
        return

    left, right = st.columns([1.65, 1])

    with left:
        if frame is not None and not frame.empty:
            st.plotly_chart(
                market_chart(
                    frame,
                    selected,
                    ["EMA9", "EMA21", "SMA20", "SMA50", "SMA200", "EMA255"],
                    rsi_col="RSI14",
                    days=252,
                    cross_columns=[
                        c for c in ["Cross9_21", "Cross20_50", "Cross50_200"]
                        if c in frame.columns
                    ],
                    rsi_lines=[(30, "RSI 30"), (50, "RSI 50"), (70, "RSI 70")],
                ),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
        else:
            st.info("No indicator history is available for the selected stock.")

    with right:
        st.subheader("Current terminal snapshot")
        st.caption("Yahoo Finance fundamentals are fetched on demand for questions about valuation, business quality, growth, profitability, debt, cash flow, or longer-term investing.")
        preview = pd.DataFrame(
            {
                "Metric": list(context["current_terminal_snapshot"].keys()),
                "Value": list(context["current_terminal_snapshot"].values()),
            }
        )
        st.dataframe(preview, use_container_width=True, height=520, hide_index=True)

    strategy_status = context.get("strategy_status", {})
    active_memberships = [
        ("Final Buy List", strategy_status.get("final_buy_list", {})),
        ("Emerging Buy List", strategy_status.get("emerging_buy_list", {})),
        ("Confluence", strategy_status.get("confluence", {})),
        ("Emerging Setups", strategy_status.get("emerging_setups", {})),
    ]
    active_memberships = [
        (name, item) for name, item in active_memberships
        if item.get("member") and item.get("verified")
    ]
    if active_memberships:
        labels = "  •  ".join(name for name, _ in active_memberships)
        st.success(f"Current verified strategy membership: {labels}")

    chat_key = f"nifty_ai_chat::{selected}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.divider()
    st.subheader(f"Chat about {selected}")

    quick_prompts = [
        ("Why is it here?", "Tell me exactly which current list or strategy this stock belongs to and explain in simple language why it qualified."),
        ("What does it mean?", "Explain what the current result means for a retail investor. Keep it simple and avoid technical jargon."),
        ("Biggest risk", "What is the biggest reason to be careful with this stock right now? Explain simply."),
        ("What next?", "What should a retail investor monitor next before becoming more confident about this setup?"),
    ]

    selected_quick = None
    cols = st.columns(4)
    for col, (label, prompt) in zip(cols, quick_prompts):
        with col:
            if st.button(label, key=f"ai_quick::{selected}::{label}", use_container_width=True):
                selected_quick = prompt

    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_prompt = st.chat_input(f"Ask Nifty AI about {selected}...")
    prompt = selected_quick or typed_prompt

    if prompt:
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        st.session_state[chat_key] = st.session_state[chat_key][-20:]
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysing current terminal data..."):
                try:
                    answer = _gemini_reply(
                        prompt,
                        context,
                        chart_png,
                        st.session_state[chat_key][:-1],
                    )
                    st.markdown(answer)
                    st.session_state[chat_key].append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state[chat_key] = st.session_state[chat_key][-20:]
                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

    st.caption(
        f"Model: {GEMINI_MODEL}. AI is called only when you submit a question. "
        f"Technical chart context is included by default, while Yahoo Finance fundamentals are fetched only when relevant. "
        f"Strategy verification uses the {AI_STRATEGY_PREFILTER_SCORE}+ high-conviction candidate pool and does not alter the terminal's visible strategy rules."
    )


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
    "Discover": [
        st.Page(
            emerging_setups_page,
            title="Emerging Setups",
            icon="🌱",
            url_path="emerging-setups",
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
    "AI": [
        st.Page(
            nifty_ai_page,
            title="Nifty AI Analyst",
            icon="🤖",
            url_path="nifty-ai",
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
    position="sidebar",
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
<b>5.</b> Open Final Buy List<br>
<b>6.</b> Ask Nifty AI Analyst why a shortlisted stock qualified<br>
<b>Optional.</b> Check Emerging Setups for newer stocks
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
