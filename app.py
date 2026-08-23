
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
    --bg: #080a0d;
    --panel: #0d1116;
    --panel-2: #11161d;
    --border: #252c35;
    --text: #e9eef3;
    --muted: #8c98a6;
    --accent: #f0a51a;
    --green: #26c281;
    --red: #ef6461;
    --blue: #6ea8fe;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: Helvetica, Arial, sans-serif;
}

html, body, [class*="css"] {
    font-family: Helvetica, Arial, sans-serif;
}

[data-testid="stSidebar"] {
    background: #0a0d11;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    font-family: Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.terminal-topbar {
    background: #0b0e12;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    margin-bottom: 12px;
}

.terminal-brand {
    font-size: 23px;
    font-weight: 700;
    letter-spacing: .4px;
}

.terminal-kicker {
    color: var(--accent);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.terminal-sub {
    color: var(--muted);
    font-size: 12px;
    margin-top: 3px;
}

.section-kicker {
    color: var(--accent);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 700;
    margin-top: 8px;
}

.hero {
    padding: 20px 0 14px 0;
}

.hero-title {
    font-size: 34px;
    line-height: 1.04;
    font-weight: 700;
    margin: 4px 0;
}

.hero-copy {
    max-width: 820px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
}

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 16px;
}

.card-title {
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
}

.card-copy {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    padding: 12px 14px;
}

.metric-label {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    margin-top: 2px;
}

.metric-note {
    color: var(--muted);
    font-size: 10px;
    margin-top: 2px;
}

.signal-green {
    color: var(--green);
    font-weight: 700;
}

.signal-red {
    color: var(--red);
    font-weight: 700;
}

.small-note {
    color: var(--muted);
    font-size: 11px;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    padding: 8px 12px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 2px;
    border: 1px solid #4b5563;
    background: #12171e;
    color: var(--text);
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent);
    background: #171d25;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
}

div[data-baseweb="tab-list"] {
    gap: 2px;
}

button[data-baseweb="tab"] {
    background: #0b0f14;
    border: 1px solid var(--border);
    color: var(--muted);
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text);
    border-bottom: 2px solid var(--accent);
}

hr {
    border-color: var(--border);
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------

def terminal_header(page_title: str, subtitle: str):
    universe_count = (
        len(st.session_state.get("universe", []))
        if "universe" in st.session_state
        else 0
    )
    scan_date = st.session_state.get("scan_date", "NOT RUN")

    st.markdown(
        f"""
<div class="terminal-topbar">
  <div class="terminal-kicker">NSE EQUITY RESEARCH TERMINAL</div>
  <div class="terminal-brand">NIFTY MARKET TERMINAL</div>
  <div class="terminal-sub">
      {page_title} · {subtitle} · UNIVERSE {universe_count or "—"}
      · LAST SCAN {scan_date}
  </div>
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
        "A simple workflow for finding structured equity setups",
    )

    st.markdown(
        """
<div class="hero">
  <div class="section-kicker">Market research workspace</div>
  <div class="hero-title">Find. Filter. Validate.</div>
  <div class="hero-copy">
    Scan India's broad listed-equity universe, separate market regime from
    momentum and pullback conditions, then use convergence to create a short
    research list. The terminal is designed as a decision-support tool, not
    an automatic trading system.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if "universe" not in st.session_state:
        ucount = "—"
    else:
        ucount = f"{len(st.session_state['universe']):,}"

    snapshot = st.session_state.get("snapshot", pd.DataFrame())
    conv = st.session_state.get("convergence", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UNIVERSE", ucount)
    c2.metric("STOCKS SCANNED", f"{len(snapshot):,}")
    c3.metric(
        "BULLISH REGIME",
        (
            f"{int(snapshot['BullRegime'].sum()):,}"
            if not snapshot.empty
            else "—"
        ),
    )
    c4.metric(
        "HIGH-CONVICTION",
        (
            f"{int((conv['ConvergenceScore'] >= 65).sum()):,}"
            if not conv.empty
            else "—"
        ),
    )

    st.markdown('<div class="section-kicker">Research modules</div>', unsafe_allow_html=True)

    modules = st.columns(4)

    with modules[0]:
        card(
            "01 · MARKET REGIME",
            "Is the long-term structure supportive? Uses 50/200 SMA and price location.",
        )

    with modules[1]:
        card(
            "02 · MOMENTUM",
            "Short-term direction and fresh 9/21 EMA crossovers.",
        )

    with modules[2]:
        card(
            "03 · SWING STRUCTURE",
            "Medium-term trend alignment using 20/50 SMA.",
        )

    with modules[3]:
        card(
            "04 · PULLBACK",
            "Potential oversold pullbacks near EMA 255.",
        )

    st.markdown('<div class="section-kicker">How to use</div>', unsafe_allow_html=True)

    steps = st.columns(4)
    for i, (title, copy) in enumerate(
        [
            ("1. Scan", "Download one shared price dataset for the whole universe."),
            ("2. Explore", "Review each strategy as a separate market dimension."),
            ("3. Converge", "Use Trend Score + Entry Score rather than double-counting indicators."),
            ("4. Shortlist", "Only finalists receive slower fundamental enrichment."),
        ]
    ):
        with steps[i]:
            card(title, copy)

    st.caption(
        "Data source: NSE constituent files + Yahoo Finance daily prices. "
        "Signals are research candidates and are not investment advice."
    )


def scan_page():
    terminal_header(
        "Scan Engine",
        "One shared download. All strategies reuse the result.",
    )

    st.markdown(
        '<div class="section-kicker">Market data controls</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])

    with c1:
        universe_name = st.selectbox(
            "Universe",
            ["NIFTY TOTAL MARKET", "NIFTY 500"],
            index=0,
        )

    with c2:
        history_years = st.selectbox(
            "History",
            [3, 4, 5],
            index=1,
        )

    with c3:
        batch_size = st.select_slider(
            "Batch size",
            options=[50, 75, 100],
            value=75,
        )

    with c4:
        if st.button(
            "RUN MARKET SCAN",
            type="primary",
            use_container_width=True,
        ):
            run_scan = True
        else:
            run_scan = False

    if run_scan:
        try:
            with st.spinner("Loading current universe..."):
                universe = load_universe(universe_name)

            progress = st.progress(0)
            status = st.empty()

            def update(batch, total, failures):
                progress.progress(batch / total)
                status.write(
                    f"Downloading batch {batch}/{total} · "
                    f"unresolved {failures}"
                )

            prices, failures = download_prices(
                universe,
                years=history_years,
                batch_size=batch_size,
                progress_callback=update,
            )

            if prices.empty:
                st.error("No price data returned.")
                st.stop()

            status.write("Calculating all indicators once...")
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
            st.session_state["scan_date"] = (
                datetime.now().strftime("%Y-%m-%d %H:%M")
            )

            progress.empty()
            status.empty()

            st.success(
                f"Scan complete. {len(snapshot):,} stocks processed."
            )

        except Exception as exc:
            st.error("The market scan failed.")
            with st.expander("Technical details"):
                st.exception(exc)

    if "snapshot" in st.session_state:
        snapshot = st.session_state["snapshot"]
        failures = st.session_state.get("failures", [])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CONSTITUENTS", f"{len(st.session_state['universe']):,}")
        m2.metric("PRICE SERIES", f"{len(snapshot):,}")
        m3.metric(
            "BULLISH REGIME",
            f"{int(snapshot['BullRegime'].sum()):,}",
        )
        m4.metric(
            "PULLBACK SETUPS",
            f"{int(snapshot['Pullback'].sum()):,}",
        )

        if failures:
            st.warning(
                f"{len(failures)} symbols remained unresolved after retry."
            )

        st.markdown('<div class="section-kicker">Data quality</div>', unsafe_allow_html=True)

        q1, q2, q3 = st.columns(3)
        q1.metric(
            "Coverage",
            f"{len(snapshot) / max(len(st.session_state['universe']), 1) * 100:.1f}%",
        )
        q2.metric(
            "Latest observation",
            pd.to_datetime(snapshot["Date"]).max().strftime("%d %b %Y"),
        )
        q3.metric(
            "Stored history",
            f"{len(st.session_state['prices']):,} daily rows",
        )

        st.download_button(
            "EXPORT SNAPSHOT CSV",
            data=snapshot.to_csv(index=False).encode(),
            file_name="nifty_total_market_snapshot.csv",
            mime="text/csv",
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
    "Overview": [
        st.Page(
            home_page,
            title="Home",
            icon="🏠",
            url_path="home",
            default=True,
        ),
    ],
    "Workflow": [
        st.Page(
            scan_page,
            title="Scan Engine",
            icon="🔄",
            url_path="scan-engine",
        ),
    ],
    "Strategies": [
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
            icon="📈",
            url_path="swing-20-50",
        ),
        st.Page(
            pullback_page,
            title="EMA 255 Pullback",
            icon="↔️",
            url_path="ema-255-pullback",
        ),
    ],
    "Decision": [
        st.Page(
            convergence_page,
            title="Convergence",
            icon="🎯",
            url_path="convergence",
        ),
        st.Page(
            buying_list_page,
            title="Final Buying List",
            icon="⭐",
            url_path="buying-list",
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
<div class="section-kicker">Terminal controls</div>
<div class="small-note">
Use Scan Engine once. Every strategy page reads the shared cached dataset.
</div>
""",
        unsafe_allow_html=True,
    )

    if "snapshot" in st.session_state:
        st.success("SCAN READY")
    else:
        st.warning("NO ACTIVE SCAN")

    st.divider()

    st.caption(
        "Built for research. Public market data. "
        "No broker API key required."
    )

pg.run()
