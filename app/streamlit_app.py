"""Hosted demo (docs/STEPS.md, Step 9).

Reads only the precomputed files in app/data/ (python -m src.app_data writes them).
Pages: this week's ranking, model trust over time, warning status, backtest,
crisis explorer.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).parent / "data"
MODELS = ["ridge", "lgbm", "mlp"]
MAIN_METHODS = ["selector", "selector_warning", "simple_avg", "ridge", "lgbm", "mlp", "ew_universe", "tasi"]


@st.cache_data
def load() -> dict:
    """Read every precomputed file once."""
    return {
        "rankings": pd.read_parquet(DATA / "rankings.parquet"),
        "weights": pd.read_csv(DATA / "selector_weights.csv", parse_dates=["week_end"]),
        "warning": pd.read_csv(DATA / "warning.csv", parse_dates=["week_end"]),
        "backtest": pd.read_parquet(DATA / "backtest.parquet"),
        "crisis": pd.read_csv(DATA / "crisis_windows.csv"),
        "crisis_dates": pd.read_csv(DATA / "crisis_dates.csv", parse_dates=["start", "end"]),
    }


def shade(fig: go.Figure, crisis_dates: pd.DataFrame) -> go.Figure:
    """Grey bands for the crisis windows."""
    for _, c in crisis_dates.iterrows():
        fig.add_vrect(x0=c["start"], x1=c["end"], fillcolor="grey", opacity=0.15, line_width=0)
    return fig


def page_ranking(d: dict) -> None:
    """Top 10% of the selector ranking for a chosen test week."""
    r = d["rankings"]
    weeks = sorted(r["week_end"].unique())
    week = st.select_slider("Test week", options=weeks, value=weeks[-1], format_func=lambda w: pd.Timestamp(w).date().isoformat())
    wk = r.loc[r["week_end"] == week].sort_values("selector_score", ascending=False)
    n = max(1, round(0.10 * len(wk)))
    st.caption(f"{len(wk)} eligible stocks; the top {n} are bought (equal weight, long only).")
    cols = ["symbol", "name", "industry_group", "selector_score", *MODELS]
    st.dataframe(wk.head(n)[cols].round(3), hide_index=True, width="stretch")


def page_trust(d: dict) -> None:
    """Selector weights over time with Brent below."""
    w = d["weights"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05)
    for m in MODELS:
        fig.add_trace(go.Scatter(x=w["week_end"], y=w[f"selector_{m}"], name=m, stackgroup="w"), row=1, col=1)
    fig.add_trace(go.Scatter(x=w["week_end"], y=w["brent"], name="Brent (USD)", line={"color": "black"}), row=2, col=1)
    fig.update_yaxes(title_text="weight", range=[0, 1], row=1, col=1)
    st.plotly_chart(shade(fig, d["crisis_dates"]).update_layout(height=550), width="stretch")


def page_warning(d: dict) -> None:
    """Bad-week probability, disagreement and exposure for a chosen week."""
    w = d["warning"]
    week = st.select_slider("Week", options=list(w["week_end"]), value=w["week_end"].iloc[-1],
                            format_func=lambda x: x.date().isoformat())
    row = w.loc[w["week_end"] == week].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Bad-week probability", f"{row['p_bad']:.2f}" if pd.notna(row["p_bad"]) else "n/a",
              help=f"threshold {row['p_bad_thr']:.2f}" if pd.notna(row["p_bad_thr"]) else None)
    c2.metric("Disagreement", f"{row['disagreement']:.2f}",
              help=f"threshold {row['disagreement_thr']:.2f}" if pd.notna(row["disagreement_thr"]) else None)
    c3.metric("Exposure", f"{row['exposure']:.0%}")
    fig = px.line(w, x="week_end", y="exposure", title="Exposure over time")
    st.plotly_chart(shade(fig, d["crisis_dates"]), width="stretch")


def page_backtest(d: dict) -> None:
    """Cumulative net value of all methods for a chosen cost level."""
    b = d["backtest"]
    cost = st.radio("One-way cost (bps)", sorted(b["cost_bps"].unique()), index=1, horizontal=True)
    methods = st.multiselect("Methods", sorted(b["method"].unique()), default=[m for m in MAIN_METHODS if m in set(b["method"])])
    x = b.loc[(b["cost_bps"] == cost) & b["method"].isin(methods)].sort_values("week_end").copy()
    x["value"] = x.groupby("method")["net"].transform(lambda r: (1 + r.fillna(0)).cumprod())
    fig = px.line(x, x="week_end", y="value", color="method", title=f"Cumulative net value, {cost} bps")
    st.plotly_chart(shade(fig, d["crisis_dates"]), width="stretch")


def page_crisis(d: dict) -> None:
    """Metrics of every method inside a chosen crisis window."""
    c = d["crisis"]
    window = st.selectbox("Crisis window", sorted(c["group"].unique()))
    cols = ["method", "mean_ic", "cum_return", "sharpe", "max_drawdown", "mean_exposure"]
    st.dataframe(c.loc[c["group"] == window, [k for k in cols if k in c.columns]].round(4), hide_index=True, width="stretch")


PAGES = {"This week's ranking": page_ranking, "Model trust over time": page_trust, "Warning status": page_warning,
         "Backtest": page_backtest, "Crisis explorer": page_crisis}


def main() -> None:
    """Sidebar navigation between the five pages."""
    st.set_page_config(page_title="Ranking Saudi stocks", layout="wide")
    st.title("Ranking Saudi stocks with machine learning")
    st.caption("Oil-aware model switching and failure warning on the Tadawul main market. "
               "Out-of-sample test weeks only; research demo, not investment advice.")
    page = st.sidebar.radio("Page", list(PAGES))
    PAGES[page](load())


main()
