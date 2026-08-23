
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

    technical = investor_quality_gate(df)

    if use_liquidity and not technical.empty:
        technical = technical.loc[
            technical["LiquidityEligible"].fillna(False)
        ].copy()

    if technical.empty:
        st.info(
            "No candidates currently pass the strict investor-quality gate. "
            "That is acceptable. The model does not force a shortlist."
        )
        return

    # Infrastructure cap only. This is NOT a cap on the number of final picks.
    # Fundamentals are the expensive/public-data step and are fetched only for
    # the strongest technical candidates to protect Streamlit Cloud limits.
    fundamental_fetch_limit = 25
    finalists_for_fundamentals = technical.head(
        fundamental_fetch_limit
    ).copy()

    rows = []
    with st.spinner(
        f"Fetching fundamentals for {len(finalists_for_fundamentals)} technical finalists..."
    ):
        for _, row in finalists_for_fundamentals.iterrows():
            fund = fundamental_snapshot(row["Yahoo Symbol"])

            fundamental_score = 0.0

            market_cap = pd.to_numeric(
                fund.get("Market Cap"),
                errors="coerce",
            )
            revenue_growth = pd.to_numeric(
                fund.get("Revenue Growth %"),
                errors="coerce",
            )
            margin = pd.to_numeric(
                fund.get("Profit Margin %"),
                errors="coerce",
            )
            debt_equity = pd.to_numeric(
                fund.get("Debt/Equity"),
                errors="coerce",
            )
            pe = pd.to_numeric(
                fund.get("PE"),
                errors="coerce",
            )

            # Missing public fundamentals are neutral, not an automatic fail.
            fundamental_score += (
                2 if pd.notna(market_cap) and market_cap >= 5_00_00_00_000 else 0
            )
            fundamental_score += (
                2 if pd.notna(revenue_growth) and revenue_growth > 0 else 0
            )
            fundamental_score += (
                2 if pd.notna(margin) and margin > 0 else 0
            )
            fundamental_score += (
                2 if pd.notna(debt_equity) and 0 <= debt_equity <= 2.5 else 0
            )
            fundamental_score += (
                2 if pd.notna(pe) and 0 < pe <= 60 else 0
            )

            investor_conviction = (
                float(row["InvestorTechnicalScore"]) * 0.90
                + fundamental_score
            )

            rows.append(
                {
                    "Symbol": row["Symbol"],
                    "Company": row["Company"],
                    "Setup": row["Setup"],
                    "Investor Conviction": round(investor_conviction, 1),
                    "Technical Quality": round(
                        float(row["InvestorTechnicalScore"]), 1
                    ),
                    "Confluence": int(row["ConvergenceScore"]),
                    "RS 3M %ile": (
                        round(float(row["RS3MPct"]), 1)
                        if pd.notna(row["RS3MPct"])
                        else None
                    ),
                    "RS 6M %ile": (
                        round(float(row["RS6MPct"]), 1)
                        if pd.notna(row["RS6MPct"])
                        else None
                    ),
                    "Volume x": (
                        round(float(row["VolumeRatio"]), 2)
                        if pd.notna(row["VolumeRatio"])
                        else None
                    ),
                    "Liquidity": row["LiquidityBucket"],
                    "RSI": (
                        round(float(row["RSI14"]), 2)
                        if pd.notna(row["RSI14"])
                        else None
                    ),
                    "ATR %": (
                        round(float(row["ATRPercent"]), 2)
                        if pd.notna(row["ATRPercent"])
                        else None
                    ),
                    "Gap %": (
                        round(float(row["GapPct"]), 2)
                        if pd.notna(row["GapPct"])
                        else None
                    ),
                    "P/E": fund["PE"],
                    "Revenue Growth %": fund["Revenue Growth %"],
                    "Net Profit Margin %": fund["Profit Margin %"],
                    "Debt/Equity": fund["Debt/Equity"],
                    "EV/EBITDA": fund["EV/EBITDA"],
                    "Market Cap": fund["Market Cap"],
                    "Yahoo Symbol": row["Yahoo Symbol"],
                }
            )

    final = pd.DataFrame(rows)

    if final.empty:
        st.info("No candidates were enriched successfully.")
        return

    final["Investor Conviction"] = pd.to_numeric(
        final["Investor Conviction"],
        errors="coerce",
    )
    final["Technical Quality"] = pd.to_numeric(
        final["Technical Quality"],
        errors="coerce",
    )

    # Score threshold, not stock count, determines the final list.
    final = (
        final.loc[
            final["Investor Conviction"] >= min_score
        ]
        .sort_values(
            ["Investor Conviction", "Technical Quality", "RS 3M %ile"],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    if final.empty:
        st.info(
            "No technically and fundamentally strong candidates cleared "
            f"the Investor Conviction threshold of {min_score:.0f}."
        )
        return

    final.insert(0, "Rank", range(1, len(final) + 1))

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

    def numeric_series(frame, column):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(float("nan"), index=frame.index)

    # -----------------------------
    # Technical score: 80 points.
    # -----------------------------
    score = pd.Series(0.0, index=working.index)

    def add_bool_score(column, points):
        nonlocal score
        if column in working.columns:
            values = working[column].fillna(False).astype(bool).astype(float)
            score += values * points

    add_bool_score("BullMomentum", 15)
    add_bool_score("BullSwing", 15)
    add_bool_score("MomentumFresh", 10)
    add_bool_score("VolumeConfirmedMomentum", 10)
    add_bool_score("Breakout20", 10)

    rs3 = numeric_series(working, "RS3MPct").clip(0, 100)
    score += rs3.fillna(0) * 0.15

    rsi = numeric_series(working, "RSI14")
    rsi_points = pd.Series(0.0, index=working.index)
    rsi_points.loc[rsi.between(50, 70, inclusive="both")] = 5.0
    rsi_points.loc[rsi.between(45, 49.999999, inclusive="both")] = 3.0
    rsi_points.loc[rsi.between(40, 44.999999, inclusive="both")] = 1.0
    # A moderately high RSI can still be constructive, but an extreme RSI
    # receives no bonus because the screen is intended to avoid chasing.
    rsi_points.loc[rsi.between(70.000001, 75, inclusive="both")] = 3.0
    score += rsi_points

    working["Technical Score"] = score.clip(0, 80).round(1)

    # -----------------------------------------
    # Fundamentals: up to 20 points.
    # -----------------------------------------
    # Fundamental data is the expensive public-data step. Fetch it only for
    # the strongest technical candidates, then cache it for this scan session.
    working["Fundamental Score"] = 0.0
    working["Fundamental Coverage"] = 0
    working["Revenue Growth %"] = float("nan")
    working["Profit Margin %"] = float("nan")
    working["Debt/Equity"] = float("nan")
    working["P/E"] = float("nan")
    working["EV/EBITDA"] = float("nan")
    working["Market Cap"] = float("nan")

    technical_order = working.sort_values(
        ["Technical Score", "RS3MPct", "VolumeRatio"],
        ascending=False,
        na_position="last",
    )
    fundamental_fetch_limit = 25
    fundamental_candidates = technical_order.head(fundamental_fetch_limit)

    cache = st.session_state.setdefault("emerging_fundamentals_cache", {})
    scan_key = str(st.session_state.get("scan_date", "current"))

    def get_fundamentals(yahoo_symbol):
        cache_key = f"{scan_key}|{yahoo_symbol}"
        if cache_key not in cache:
            try:
                result = fundamental_snapshot(yahoo_symbol)
                cache[cache_key] = result if isinstance(result, dict) else {}
            except Exception:
                cache[cache_key] = {}
        return cache[cache_key]

    with st.spinner(
        f"Checking public fundamentals for up to {len(fundamental_candidates)} emerging candidates..."
    ):
        for idx, row in fundamental_candidates.iterrows():
            fund = get_fundamentals(row.get("Yahoo Symbol"))

            market_cap = pd.to_numeric(fund.get("Market Cap"), errors="coerce")
            revenue_growth = pd.to_numeric(fund.get("Revenue Growth %"), errors="coerce")
            margin = pd.to_numeric(fund.get("Profit Margin %"), errors="coerce")
            debt_equity = pd.to_numeric(fund.get("Debt/Equity"), errors="coerce")
            pe = pd.to_numeric(fund.get("PE"), errors="coerce")
            ev_ebitda = pd.to_numeric(fund.get("EV/EBITDA"), errors="coerce")

            # 5 pts growth. Positive growth is useful; stronger growth earns more.
            growth_points = (
                5.0 if pd.notna(revenue_growth) and revenue_growth >= 15 else
                3.5 if pd.notna(revenue_growth) and revenue_growth > 0 else
                1.5 if pd.notna(revenue_growth) and revenue_growth > -10 else
                0.0
            )
            # 5 pts profitability. Positive margin is preferable to losses.
            margin_points = (
                5.0 if pd.notna(margin) and margin >= 15 else
                3.5 if pd.notna(margin) and margin > 0 else
                1.0 if pd.notna(margin) and margin > -5 else
                0.0
            )
            # 4 pts leverage. Lower debt/equity is preferred, while missing data is neutral.
            leverage_points = (
                4.0 if pd.notna(debt_equity) and 0 <= debt_equity <= 1 else
                3.0 if pd.notna(debt_equity) and debt_equity <= 2 else
                1.5 if pd.notna(debt_equity) and debt_equity <= 3 else
                0.0
            )
            # 3 pts P/E. Positive and not excessively high gets the strongest score.
            pe_points = (
                3.0 if pd.notna(pe) and 0 < pe <= 30 else
                2.0 if pd.notna(pe) and pe <= 50 else
                1.0 if pd.notna(pe) and pe <= 75 else
                0.0
            )
            # 3 pts EV/EBITDA. Same principle, with a wider range for growth stocks.
            ev_points = (
                3.0 if pd.notna(ev_ebitda) and 0 < ev_ebitda <= 20 else
                2.0 if pd.notna(ev_ebitda) and ev_ebitda <= 30 else
                1.0 if pd.notna(ev_ebitda) and ev_ebitda <= 50 else
                0.0
            )

            fundamental_score = growth_points + margin_points + leverage_points + pe_points + ev_points
            values = {
                "Revenue Growth %": revenue_growth,
                "Profit Margin %": margin,
                "Debt/Equity": debt_equity,
                "P/E": pe,
                "EV/EBITDA": ev_ebitda,
                "Market Cap": market_cap,
            }
            coverage = sum(pd.notna(v) for v in values.values())
            # Market Cap is informative but not a quality point. Coverage counts
            # the five scored fields plus market cap for transparency.
            for key, value in values.items():
                working.at[idx, key] = value
            working.at[idx, "Fundamental Score"] = round(min(fundamental_score, 20.0), 1)
            working.at[idx, "Fundamental Coverage"] = coverage

    working["Emerging Score"] = (
        working["Technical Score"] + working["Fundamental Score"]
    ).clip(0, 100).round(1)

    def setup_label(row):
        momentum = bool(row.get("BullMomentum", False))
        swing = bool(row.get("BullSwing", False))
        fresh = bool(row.get("MomentumFresh", False))
        breakout = bool(row.get("Breakout20", False))
        volume = bool(row.get("VolumeConfirmedMomentum", False))
        if breakout and volume and momentum:
            return "Trend + Volume Breakout"
        if momentum and swing and fresh:
            return "Fresh Momentum + Swing"
        if momentum and swing:
            return "Momentum + Swing"
        if breakout:
            return "Breakout Watch"
        if momentum:
            return "Momentum Watch"
        if swing:
            return "Swing Watch"
        return "Early Watch"

    working["Setup"] = working.apply(setup_label, axis=1)

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
            "Fundamental Coverage": st.column_config.NumberColumn("Fund. Coverage", format="%d/6"),
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

    buy = working.copy()
    buy = buy.loc[buy["Emerging Score"] >= 75].copy()
    buy = buy.loc[buy["Technical Score"] >= 58].copy()
    buy = buy.loc[numeric_series(buy, "RS3MPct") >= 70].copy()
    buy = buy.loc[
        buy.get("BullMomentum", pd.Series(False, index=buy.index)).fillna(False).astype(bool)
        | buy.get("BullSwing", pd.Series(False, index=buy.index)).fillna(False).astype(bool)
    ].copy()
    buy = buy.loc[buy["Fundamental Coverage"] >= 3].copy()
    buy = buy.loc[buy["Fundamental Score"] >= 9].copy()
    buy = buy.loc[numeric_series(buy, "AvgTradedValue20").fillna(0) >= 1_00_00_000].copy()

    buy = buy.sort_values(
        ["Emerging Score", "Fundamental Score", "Technical Score", "RS3MPct"],
        ascending=False,
        na_position="last",
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
                "Fundamental Coverage": st.column_config.NumberColumn("Fund. Coverage", format="%d/6"),
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
