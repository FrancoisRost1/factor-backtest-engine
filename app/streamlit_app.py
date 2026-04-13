"""
streamlit_app.py | Factor Model + Backtesting Engine Dashboard

Loads pre-computed results from outputs/backtest_results.csv (summary) and
optional time-series CSVs (outputs/ic_series_*.csv, outputs/quintile_*.csv).

Layout (per CLAUDE.md standards, quant research order):
  TAB 1: OVERVIEW             | filters, best-combination panel, rankings table
  TAB 2: FACTOR ANALYSIS      | Signal Quality, Portfolio Construction, Performance
  TAB 3: PORTFOLIO COMPARISON | long-only vs long/short side-by-side
  TAB 4: METHODOLOGY          | static research write-up

Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from style_inject import (
    inject_styles,
    styled_header,
    styled_kpi,
    styled_card,
    styled_divider,
    styled_section_label,
    apply_plotly_theme,
    TOKENS,
)

# Page config

st.set_page_config(
    page_title="Factor Backtest Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_styles()

# Constants

FACTOR_LABELS = {
    "value":          "Value (EY)",
    "momentum":       "Momentum (12-1)",
    "quality":        "Quality (ROE)",
    "size":           "Size (Log MCap)",
    "low_volatility": "Low Vol (60d sigma)",
    "composite":      "Composite",
}

# Price-based factors have no point-in-time data issue; fundamental factors do.
FUNDAMENTAL_FACTORS = {"value", "quality", "size"}

QUINTILE_COLORS = [
    TOKENS["accent_danger"],
    TOKENS["accent_warning"],
    TOKENS["text_secondary"],
    TOKENS["accent_primary"],
    TOKENS["accent_success"],
]

# Helpers

def pct(v, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v * 100:.{dec}f}%"

def fmt_alpha(v):
    """Format annualised alpha, collapsing near-zero to 'approx 0.0%' to avoid '-0.0%'."""
    v = _safe(v)
    if np.isnan(v):
        return ""
    if abs(v) < 0.0005:
        return "~ 0.0%"
    return f"{v * 100:.1f}%"

def fmt3(v):
    """Format to 3 decimal places (used for Mean IC and IC IR in tables)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.3f}"

def fmt2(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.2f}"

def _safe(v):
    """Return np.nan if v is missing, else float(v)."""
    try:
        f = float(v)
        return np.nan if np.isnan(f) else f
    except Exception:
        return np.nan

def delta_color(v, good_above=0.0):
    v = _safe(v)
    if np.isnan(v):
        return TOKENS["text_muted"]
    return TOKENS["accent_success"] if v > good_above else TOKENS["accent_danger"]

def styled_kpi_colored(label, value, v_for_color, good_above=0.0, delta=""):
    """KPI whose delta text is the value itself, colored by sign vs threshold."""
    dc = delta_color(v_for_color, good_above)
    styled_kpi(label, value, delta=delta, delta_color=dc)

def missing_series_note(label="Per-period data"):
    styled_card(
        f"<b style='color:{TOKENS['text_primary']}'>{label} not available.</b> "
        f"Run backtest with <code>save_timeseries=True</code> to enable this chart.",
        accent_color=TOKENS["accent_warning"],
    )

def interpretation_box():
    rows = [
        ("IC > 0",            TOKENS["accent_success"], "Predictive signal: factor rank correlates with forward returns"),
        ("IC ~ 0",            TOKENS["text_muted"],     "Weak or absent signal: factor has no predictive power"),
        ("IC < 0",            TOKENS["accent_danger"],  "Inverted or biased signal: ranking predicts opposite direction"),
        ("IC IR > 0.5",       TOKENS["accent_success"], "Consistent signal: mean(IC)/std(IC) often considered strong (Grinold & Kahn)"),
        ("Monotonic Q1 to Q5", TOKENS["accent_success"], "Strong factor: returns increase as predicted across all quintiles"),
        ("Flat quintiles",    TOKENS["text_muted"],     "No factor signal: ranking explains nothing about returns"),
    ]
    table_rows = "".join(
        f"<tr>"
        f"<td style='padding:3px 8px;color:{color};font-weight:600;white-space:nowrap'>{label}</td>"
        f"<td style='padding:3px 8px;color:{TOKENS['text_secondary']}'>{desc}</td>"
        f"</tr>"
        for (label, color, desc) in rows
    )
    body = (
        f"<div style='font-size:0.65rem;font-weight:700;color:{TOKENS['accent_primary']};"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>How to Interpret</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.75rem'>{table_rows}</table>"
    )
    styled_card(body, accent_color=TOKENS["accent_primary"])

# Data loading

OUTPUTS = Path(__file__).parent.parent / "outputs"

@st.cache_data
def load_results() -> Optional[pd.DataFrame]:
    path = OUTPUTS / "backtest_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_ic_series(factor: str, freq: str) -> Optional[pd.DataFrame]:
    """Load per-period IC values if exported by main.py."""
    path = OUTPUTS / f"ic_{factor}_{freq}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

@st.cache_data
def load_quintile_returns(factor: str, freq: str, wt: str) -> Optional[pd.DataFrame]:
    """Load per-period quintile returns if exported by main.py."""
    path = OUTPUTS / f"quintile_{factor}_{freq}_{wt}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

# Header

styled_header(
    "Factor Backtest Engine",
    "S&P 500 | 5 Factors | 2016-2026 | 60 Combinations",
)

styled_card(
    f"<b style='color:{TOKENS['accent_warning']}'>Research Limitation: Results Not Investable.</b> "
    "Fundamental factors (Value, Quality, Size) use non-point-in-time data: yfinance "
    "current values applied retroactively across the entire backtest period. Returns for "
    "these factors are likely overstated. Price-based factors (Momentum, Low Volatility) are clean.",
    accent_color=TOKENS["accent_warning"],
)

df_all = load_results()

def _no_data_banner():
    styled_card(
        f"<b style='color:{TOKENS['accent_warning']}'>No results found.</b> "
        f"Run the backtest pipeline first: <code>python main.py</code>. "
        f"This fetches data from Yahoo Finance, runs all 60 backtest combinations, "
        f"and saves results to <code>outputs/backtest_results.csv</code>.",
        accent_color=TOKENS["accent_warning"],
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "OVERVIEW",
    "FACTOR ANALYSIS",
    "PORTFOLIO COMPARISON",
    "METHODOLOGY",
])

# ======================================================================
# TAB 1 | OVERVIEW
# ======================================================================

with tab1:
    if df_all is None:
        _no_data_banner()
    else:
        # Filters
        styled_section_label("FILTERS")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            all_factors = sorted(df_all["factor"].unique())
            sel_factor = st.selectbox(
                "Factor",
                ["All"] + [FACTOR_LABELS.get(f, f) for f in all_factors],
                key="ov_factor",
            )
        with fc2:
            sel_freq = st.selectbox("Frequency",
                ["All", "monthly", "quarterly", "annual"], key="ov_freq")
        with fc3:
            sel_wt = st.selectbox("Weighting",
                ["All", "equal", "cap_weight"], key="ov_wt")
        with fc4:
            sel_pt = st.selectbox("Portfolio Type",
                ["All", "long_only", "long_short"], key="ov_pt")

        df = df_all.copy()
        if sel_factor != "All":
            raw = {v: k for k, v in FACTOR_LABELS.items()}.get(sel_factor, sel_factor)
            df = df[df["factor"] == raw]
        if sel_freq != "All":
            df = df[df["frequency"] == sel_freq]
        if sel_wt != "All":
            df = df[df["weighting"] == sel_wt]
        if sel_pt != "All":
            df = df[df["portfolio_type"] == sel_pt]

        if df.empty:
            st.info("No results match the current filters.")
        else:
            styled_divider()
            styled_section_label("BEST PERFORMANCE (BY GROSS SHARPE)")
            styled_card(
                "Selected by Sharpe ratio. May reflect risk exposure or data bias, "
                "not genuine predictive power. Check IC and quintile spread to assess signal quality.",
                accent_color=TOKENS["accent_warning"],
            )

            best = df.loc[df["sharpe_gross"].idxmax()]
            ret_v    = _safe(best.get("return_gross", np.nan))
            sharpe_v = _safe(best.get("sharpe_gross", np.nan))
            mdd_v    = _safe(best.get("max_dd", np.nan))
            alpha_v  = _safe(best.get("alpha", np.nan))
            ic_v     = _safe(best.get("mean_ic", np.nan))
            to_v     = _safe(best.get("avg_turnover", np.nan))
            ic_ir_v  = _safe(best.get("ic_ir", np.nan))

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1:
                styled_kpi("Ann. Return (Gross)", pct(ret_v), delta="", delta_color=delta_color(ret_v))
            with k2:
                styled_kpi("Sharpe (Gross)", fmt2(sharpe_v), delta="", delta_color=delta_color(sharpe_v))
            with k3:
                styled_kpi("Max Drawdown", pct(mdd_v), delta="", delta_color=delta_color(mdd_v, -0.05))
            with k4:
                styled_kpi("Alpha (Ann.)", fmt_alpha(alpha_v), delta="", delta_color=delta_color(alpha_v))
            with k5:
                styled_kpi("Mean IC", fmt2(ic_v), delta=f"IR {fmt2(ic_ir_v)}", delta_color=delta_color(ic_v))
            with k6:
                styled_kpi("Avg Turnover / Period", pct(to_v))

            st.markdown(
                f"<div style='font-size:0.75rem;color:{TOKENS['text_muted']};margin:6px 0 16px 2px'>"
                f"Combination: <b style='color:{TOKENS['text_primary']}'>"
                f"{FACTOR_LABELS.get(best['factor'], best['factor'])}</b> | "
                f"{best['frequency']} | {best['weighting']} | {best['portfolio_type']}"
                f"</div>", unsafe_allow_html=True)

            # Summary table
            styled_section_label("ALL COMBINATIONS: SORTED BY SHARPE (GROSS)")

            display_cols = {
                "factor": "Factor", "frequency": "Freq", "weighting": "Weighting",
                "portfolio_type": "Type",
                "return_gross": "Return (G)", "return_net": "Return (N)",
                "sharpe_gross": "Sharpe (G)", "sharpe_net": "Sharpe (N)",
                "sortino_gross": "Sortino", "max_dd": "Max DD",
                "calmar": "Calmar", "hit_rate": "Hit Rate",
                "alpha": "Alpha", "beta": "Beta",
                "mean_ic": "Mean IC", "ic_ir": "IC IR",
                "avg_turnover": "Turnover",
            }
            df_show = df[[c for c in display_cols if c in df.columns]].copy()
            df_show = df_show.rename(columns=display_cols)
            df_show["Factor"] = df_show["Factor"].map(lambda x: FACTOR_LABELS.get(x, x))
            df_show = df_show.sort_values("Sharpe (G)", ascending=False)
            for col in ["Return (G)", "Return (N)", "Max DD", "Turnover", "Hit Rate"]:
                if col in df_show.columns:
                    df_show[col] = df_show[col].apply(
                        lambda v: pct(_safe(v)) if not (isinstance(v, float) and np.isnan(v)) else ""
                    )
            if "Alpha" in df_show.columns:
                df_show["Alpha"] = df_show["Alpha"].apply(lambda v: fmt_alpha(_safe(v)))
            for col in ["Sharpe (G)", "Sharpe (N)", "Sortino", "Calmar", "Beta"]:
                if col in df_show.columns:
                    df_show[col] = df_show[col].apply(
                        lambda v: fmt2(_safe(v)) if not (isinstance(v, float) and np.isnan(v)) else ""
                    )
            for col in ["Mean IC", "IC IR"]:
                if col in df_show.columns:
                    df_show[col] = df_show[col].apply(
                        lambda v: fmt3(_safe(v)) if not (isinstance(v, float) and np.isnan(v)) else ""
                    )
            st.dataframe(df_show, use_container_width=True,
                         height=min(600, 38 + 35 * len(df_show)), hide_index=True)


# ======================================================================
# TAB 2 | FACTOR ANALYSIS
# Structure: Signal Quality, Portfolio Construction, Performance
# ======================================================================

with tab2:
    if df_all is None:
        _no_data_banner()
    else:
        sel_c1, sel_c2, sel_c3, sel_c4 = st.columns([2, 2, 2, 2])
        with sel_c1:
            sel_fa = st.selectbox(
                "Factor",
                sorted(df_all["factor"].unique()),
                format_func=lambda x: FACTOR_LABELS.get(x, x),
                key="fa_factor",
            )
        with sel_c2:
            sel_fa_freq = st.selectbox(
                "Primary Frequency",
                ["monthly", "quarterly", "annual"],
                key="fa_freq",
            )
        with sel_c3:
            sel_fa_wt = st.selectbox(
                "Weighting", ["equal", "cap_weight"], key="fa_wt")
        with sel_c4:
            sel_fa_pt = st.selectbox(
                "Portfolio Type", ["long_only", "long_short"], key="fa_pt")

        df_fa = df_all[
            (df_all["factor"] == sel_fa) &
            (df_all["portfolio_type"] == sel_fa_pt)
        ].copy()

        df_sel = df_fa[
            (df_fa["frequency"] == sel_fa_freq) &
            (df_fa["weighting"] == sel_fa_wt)
        ]
        row_sel = df_sel.iloc[0] if len(df_sel) > 0 else None

        if sel_fa in FUNDAMENTAL_FACTORS:
            styled_card(
                f"<b style='color:{TOKENS['accent_warning']}'>{FACTOR_LABELS.get(sel_fa, sel_fa)}</b> "
                f"uses non-point-in-time fundamental data. Signal metrics and performance "
                f"figures are likely upward-biased.",
                accent_color=TOKENS["accent_warning"],
            )

        # ----- BLOCK 1: SIGNAL QUALITY -----
        styled_divider()
        styled_section_label("1. SIGNAL QUALITY | DOES THE FACTOR PREDICT RETURNS?")

        sig_left, sig_right = st.columns([5, 2])

        with sig_left:
            styled_section_label("INFORMATION COEFFICIENT")
            ic_mean  = _safe(row_sel["mean_ic"])  if row_sel is not None else np.nan
            ic_ir    = _safe(row_sel["ic_ir"])    if row_sel is not None else np.nan

            ic_c1, ic_c2, ic_c3 = st.columns(3)
            with ic_c1:
                ic_col = (TOKENS["accent_success"] if not np.isnan(ic_mean) and ic_mean > 0.02
                          else TOKENS["accent_danger"] if not np.isnan(ic_mean) and ic_mean < 0
                          else TOKENS["accent_warning"])
                styled_kpi("Mean IC", fmt2(ic_mean), delta="", delta_color=ic_col)
            with ic_c2:
                irir_col = (TOKENS["accent_success"] if not np.isnan(ic_ir) and ic_ir > 0.5
                            else TOKENS["accent_warning"] if not np.isnan(ic_ir) and ic_ir > 0
                            else TOKENS["accent_danger"])
                styled_kpi("IC IR", fmt2(ic_ir),
                           delta="ref > 0.5", delta_color=irir_col)
            with ic_c3:
                _ic_ts_for_pct = load_ic_series(sel_fa, sel_fa_freq)
                if _ic_ts_for_pct is not None and len(_ic_ts_for_pct) > 0:
                    _col = _ic_ts_for_pct.columns[0]
                    _vals = _ic_ts_for_pct[_col].dropna()
                    if len(_vals) > 0:
                        pct_pos = float((_vals > 0).mean())
                        styled_kpi("% Periods IC > 0", pct(pct_pos),
                                   delta=f"n = {len(_vals)}",
                                   delta_color=delta_color(pct_pos, 0.5))
                    else:
                        styled_kpi("% Periods IC > 0", "n/a")
                else:
                    styled_kpi("% Periods IC > 0", "n/a",
                               delta="save_timeseries=True",
                               delta_color=TOKENS["text_muted"])

            styled_section_label("IC OVER TIME")
            ic_ts = load_ic_series(sel_fa, sel_fa_freq)
            show_rolling = st.checkbox("Show 6-period rolling mean", value=True, key="ic_rolling")

            if ic_ts is not None and len(ic_ts) > 0:
                col_name = ic_ts.columns[0] if len(ic_ts.columns) > 0 else None
                if col_name is not None:
                    ic_vals = ic_ts[col_name].dropna()
                    fig_ic = go.Figure()
                    fig_ic.add_trace(go.Bar(
                        x=ic_vals.index, y=ic_vals.values,
                        marker_color=[TOKENS["accent_success"] if v > 0 else TOKENS["accent_danger"]
                                      for v in ic_vals.values],
                        marker_line_width=0, opacity=0.85, name="IC",
                        showlegend=True,
                    ))
                    if show_rolling and len(ic_vals) >= 6:
                        roll = ic_vals.rolling(6).mean()
                        fig_ic.add_trace(go.Scatter(
                            x=roll.index, y=roll.values,
                            line=dict(color=TOKENS["accent_warning"], width=1.5),
                            name="6-period MA",
                        ))
                    fig_ic.add_hline(y=0, line_color=TOKENS["border_default"], line_width=1)
                    fig_ic.update_layout(
                        title=f"{FACTOR_LABELS.get(sel_fa, sel_fa)}: IC per Period",
                        xaxis_title="Date",
                        yaxis_title="IC (Spearman)",
                    )
                    apply_plotly_theme(fig_ic)
                    st.plotly_chart(fig_ic, use_container_width=True)
            else:
                missing_series_note("IC time-series")

            styled_section_label("QUINTILE RETURN SPREAD (Q1 = WORST, Q5 = BEST)")
            qt_data = load_quintile_returns(sel_fa, sel_fa_freq, sel_fa_wt)

            if qt_data is not None and len(qt_data) > 0:
                q_means = qt_data.mean()
                quintile_labels = [f"Q{int(c)}" if str(c).isdigit() else str(c)
                                   for c in q_means.index]
                fig_qbar = go.Figure()
                fig_qbar.add_trace(go.Bar(
                    x=quintile_labels,
                    y=q_means.values * 100,
                    marker_color=QUINTILE_COLORS[:len(q_means)],
                    text=[pct(v) for v in q_means.values],
                    textposition="outside",
                    textfont=dict(size=11, color=TOKENS["text_secondary"]),
                    showlegend=False,
                ))
                fig_qbar.add_hline(y=0, line_color=TOKENS["border_default"], line_width=1)
                fig_qbar.update_layout(
                    title=f"Mean Period Return by Quintile: {sel_fa_freq.capitalize()}",
                    xaxis_title="Quintile",
                    yaxis_title="Return (%)",
                    yaxis_ticksuffix="%",
                )
                apply_plotly_theme(fig_qbar)
                st.plotly_chart(fig_qbar, use_container_width=True)

                styled_section_label("CUMULATIVE RETURN BY QUINTILE")
                log_scale = st.checkbox("Log scale", value=False, key="log_q")
                fig_qcum = go.Figure()
                for i, col in enumerate(qt_data.columns):
                    cum = (1 + qt_data[col].fillna(0)).cumprod()
                    label = f"Q{int(col)}" if str(col).isdigit() else str(col)
                    fig_qcum.add_trace(go.Scatter(
                        x=cum.index, y=cum.values,
                        name=label,
                        line=dict(color=QUINTILE_COLORS[i % 5], width=1.5),
                        mode="lines",
                    ))
                fig_qcum.update_layout(
                    title=f"Cumulative Return by Quintile: {sel_fa_freq.capitalize()}",
                    xaxis_title="Date",
                    yaxis_title="Growth of $1",
                )
                if log_scale:
                    fig_qcum.update_layout(yaxis_type="log")
                apply_plotly_theme(fig_qcum)
                st.plotly_chart(fig_qcum, use_container_width=True)
            else:
                missing_series_note("Quintile return series")

        with sig_right:
            st.markdown("<div style='margin-top:48px'></div>", unsafe_allow_html=True)
            interpretation_box()

            styled_section_label("IC BY FREQUENCY")
            for f in ["monthly", "quarterly", "annual"]:
                sub = df_fa[(df_fa["frequency"] == f) & (df_fa["weighting"] == sel_fa_wt)]
                if len(sub) > 0:
                    ic_v = _safe(sub["mean_ic"].values[0])
                    ir_v = _safe(sub["ic_ir"].values[0])
                    ic_col = (TOKENS["accent_success"] if not np.isnan(ic_v) and ic_v > 0
                              else TOKENS["accent_danger"])
                    styled_kpi(f.capitalize(), f"IC {fmt2(ic_v)}",
                               delta=f"IR {fmt2(ir_v)}", delta_color=ic_col)

        styled_divider()

        # ----- BLOCK 2: PORTFOLIO CONSTRUCTION -----
        styled_section_label("2. PORTFOLIO CONSTRUCTION | SIGNAL TO PORTFOLIO")

        pc_c1, pc_c2 = st.columns([3, 2])

        with pc_c1:
            styled_section_label("SHARPE RATIO: EQUAL vs CAP-WEIGHT BY FREQUENCY")
            freqs = ["monthly", "quarterly", "annual"]
            wts   = ["equal", "cap_weight"]
            wt_colors = {"equal": TOKENS["accent_primary"], "cap_weight": TOKENS["accent_info"]}

            fig_sh = go.Figure()
            for wt in wts:
                sub = df_fa[df_fa["weighting"] == wt]
                sh_vals = [
                    _safe(sub[sub["frequency"] == f]["sharpe_gross"].values[0])
                    if len(sub[sub["frequency"] == f]) > 0 else np.nan
                    for f in freqs
                ]
                fig_sh.add_trace(go.Bar(
                    name=wt,
                    x=[f.capitalize() for f in freqs],
                    y=sh_vals,
                    marker_color=wt_colors[wt],
                    marker_line_color=TOKENS["border_default"], marker_line_width=1,
                    text=[fmt2(v) for v in sh_vals],
                    textposition="outside",
                    textfont=dict(size=10, color=TOKENS["text_secondary"]),
                ))
            fig_sh.add_hline(y=0, line_color=TOKENS["border_default"], line_width=1)
            fig_sh.update_layout(
                title="Sharpe (Gross)",
                xaxis_title="Rebalance Frequency",
                yaxis_title="Sharpe Ratio",
                barmode="group", bargap=0.2,
            )
            apply_plotly_theme(fig_sh)
            st.plotly_chart(fig_sh, use_container_width=True)

        with pc_c2:
            styled_section_label("TURNOVER BY FREQUENCY")
            for wt_val, wt_label in [("equal", "Equal Weight"),
                                      ("cap_weight", "Cap Weight")]:
                sub_wt = df_fa[df_fa["weighting"] == wt_val]
                styled_section_label(wt_label.upper())
                to_cols = st.columns(3)
                for i, f in enumerate(freqs):
                    sub_f = sub_wt[sub_wt["frequency"] == f]
                    to_v = _safe(sub_f["avg_turnover"].values[0]) if len(sub_f) > 0 else np.nan
                    with to_cols[i]:
                        styled_kpi(f.capitalize(), pct(to_v))

        styled_section_label("EQUAL vs CAP-WEIGHT: FULL COMPARISON TABLE")
        comp_cols = ["frequency", "weighting", "return_gross", "return_net",
                     "sharpe_gross", "sharpe_net", "max_dd", "calmar",
                     "hit_rate", "avg_turnover", "mean_ic", "ic_ir"]
        comp_cols = [c for c in comp_cols if c in df_fa.columns]
        cmp = df_fa[comp_cols].copy().sort_values(["frequency", "weighting"])
        cmp = cmp.rename(columns={
            "return_gross": "Return (G)", "return_net": "Return (N)",
            "sharpe_gross": "Sharpe (G)", "sharpe_net": "Sharpe (N)",
            "max_dd": "Max DD", "calmar": "Calmar", "hit_rate": "Hit Rate",
            "avg_turnover": "Turnover", "frequency": "Freq", "weighting": "Weighting",
            "mean_ic": "Mean IC", "ic_ir": "IC IR",
        })
        for col in ["Return (G)", "Return (N)", "Max DD", "Hit Rate", "Turnover", "Mean IC"]:
            if col in cmp.columns:
                cmp[col] = cmp[col].apply(lambda v: pct(_safe(v)))
        for col in ["Sharpe (G)", "Sharpe (N)", "Calmar", "IC IR"]:
            if col in cmp.columns:
                cmp[col] = cmp[col].apply(lambda v: fmt2(_safe(v)))
        st.dataframe(cmp, use_container_width=True, hide_index=True)

        styled_divider()

        # ----- BLOCK 3: PERFORMANCE -----
        styled_section_label("3. PERFORMANCE | RETURNS, RISK-ADJUSTED METRICS, REGRESSION")

        show_net = st.checkbox("Compare gross vs net", value=True, key="show_net")

        perf_c1, perf_c2 = st.columns([3, 2])

        with perf_c1:
            styled_section_label("ANNUALISED RETURN BY FREQUENCY")
            fig_ret = go.Figure()
            for wt in wts:
                sub = df_fa[df_fa["weighting"] == wt]
                g_vals = [
                    _safe(sub[sub["frequency"] == f]["return_gross"].values[0])
                    if len(sub[sub["frequency"] == f]) > 0 else np.nan
                    for f in freqs
                ]
                n_vals = [
                    _safe(sub[sub["frequency"] == f]["return_net"].values[0])
                    if len(sub[sub["frequency"] == f]) > 0 else np.nan
                    for f in freqs
                ]
                fig_ret.add_trace(go.Bar(
                    name=f"{wt} Gross",
                    x=[f.capitalize() for f in freqs],
                    y=[v * 100 if not np.isnan(v) else None for v in g_vals],
                    marker_color=wt_colors[wt], opacity=0.9,
                    text=[pct(v) for v in g_vals],
                    textposition="outside", textfont=dict(size=9),
                ))
                if show_net:
                    fig_ret.add_trace(go.Bar(
                        name=f"{wt} Net",
                        x=[f.capitalize() for f in freqs],
                        y=[v * 100 if not np.isnan(v) else None for v in n_vals],
                        marker_color=wt_colors[wt], opacity=0.4,
                        marker_line_color=wt_colors[wt], marker_line_width=1,
                        text=[pct(v) for v in n_vals],
                        textposition="outside", textfont=dict(size=9),
                    ))
            fig_ret.add_hline(y=0, line_color=TOKENS["border_default"], line_width=1)
            fig_ret.update_layout(
                title="Annualised Return (%)",
                xaxis_title="Rebalance Frequency",
                yaxis_title="Return (%)",
                barmode="group", yaxis_ticksuffix="%",
            )
            apply_plotly_theme(fig_ret)
            st.plotly_chart(fig_ret, use_container_width=True)

        with perf_c2:
            styled_section_label("KEY PERFORMANCE METRICS")
            if row_sel is not None:
                metrics = [
                    ("Ann. Return (Gross)", pct(_safe(row_sel.get("return_gross", np.nan))),
                     delta_color(_safe(row_sel.get("return_gross", np.nan)))),
                    ("Ann. Return (Net)",   pct(_safe(row_sel.get("return_net", np.nan))),
                     delta_color(_safe(row_sel.get("return_net", np.nan)))),
                    ("Sharpe (Gross)",      fmt2(_safe(row_sel.get("sharpe_gross", np.nan))),
                     delta_color(_safe(row_sel.get("sharpe_gross", np.nan)))),
                    ("Sortino",             fmt2(_safe(row_sel.get("sortino_gross", np.nan))),
                     delta_color(_safe(row_sel.get("sortino_gross", np.nan)))),
                    ("Max Drawdown",        pct(_safe(row_sel.get("max_dd", np.nan))),
                     delta_color(_safe(row_sel.get("max_dd", np.nan)), -0.05)),
                    ("Calmar",              fmt2(_safe(row_sel.get("calmar", np.nan))),
                     delta_color(_safe(row_sel.get("calmar", np.nan)))),
                    ("Hit Rate vs SPY",     pct(_safe(row_sel.get("hit_rate", np.nan))),
                     delta_color(_safe(row_sel.get("hit_rate", np.nan)), 0.5)),
                    ("Alpha (Ann.)",        fmt_alpha(_safe(row_sel.get("alpha", np.nan))),
                     delta_color(_safe(row_sel.get("alpha", np.nan)))),
                    ("Beta vs SPY",         fmt2(_safe(row_sel.get("beta", np.nan))),
                     TOKENS["text_muted"]),
                    ("R-squared",           fmt2(_safe(row_sel.get("r_squared", np.nan))),
                     TOKENS["text_muted"]),
                ]
                m_c1, m_c2 = st.columns(2)
                for i, (lbl, val, clr) in enumerate(metrics):
                    with (m_c1 if i % 2 == 0 else m_c2):
                        styled_kpi(lbl, val, delta="", delta_color=clr)


# ======================================================================
# TAB 3 | PORTFOLIO COMPARISON
# ======================================================================

with tab3:
    if df_all is None:
        _no_data_banner()
    else:
        sc1, sc2, sc3 = st.columns([2, 2, 2])
        with sc1:
            sel_pc_f = st.selectbox("Factor",
                sorted(df_all["factor"].unique()),
                format_func=lambda x: FACTOR_LABELS.get(x, x),
                key="pc_factor")
        with sc2:
            sel_pc_freq = st.selectbox("Frequency",
                ["monthly", "quarterly", "annual"], key="pc_freq")
        with sc3:
            sel_pc_wt = st.selectbox("Weighting",
                ["equal", "cap_weight"], key="pc_wt")

        df_pc = df_all[
            (df_all["factor"] == sel_pc_f) &
            (df_all["frequency"] == sel_pc_freq) &
            (df_all["weighting"] == sel_pc_wt)
        ].copy()

        lo = df_pc[df_pc["portfolio_type"] == "long_only"]
        ls = df_pc[df_pc["portfolio_type"] == "long_short"]

        def _g(sub, col):
            return _safe(sub[col].values[0]) if len(sub) > 0 and col in sub.columns else np.nan

        styled_divider()
        styled_section_label("1. SIGNAL QUALITY (SHARED ACROSS PORTFOLIO TYPES)")
        ic_lo = _g(lo, "mean_ic")
        icir_lo = _g(lo, "ic_ir")
        sq_c1, sq_c2, sq_c3, sq_c4 = st.columns(4)
        with sq_c1:
            styled_kpi("Mean IC", fmt2(ic_lo), delta="both types",
                       delta_color=delta_color(ic_lo))
        with sq_c2:
            ic_ir_col = (TOKENS["accent_success"] if not np.isnan(icir_lo) and icir_lo > 0.5
                         else TOKENS["accent_warning"] if not np.isnan(icir_lo) and icir_lo > 0
                         else TOKENS["accent_danger"])
            styled_kpi("IC IR", fmt2(icir_lo), delta="", delta_color=ic_ir_col)
        with sq_c3:
            hr_lo = _g(lo, "hit_rate")
            styled_kpi("Hit Rate (Long-Only)", pct(hr_lo), delta="",
                       delta_color=delta_color(hr_lo, 0.5))
        with sq_c4:
            hr_ls = _g(ls, "hit_rate")
            styled_kpi("Hit Rate (L/S)", pct(hr_ls), delta="",
                       delta_color=delta_color(hr_ls, 0.5))

        styled_divider()
        styled_section_label("2. LONG-ONLY vs LONG/SHORT | Q5 ONLY vs Q5 LONG + Q1 SHORT")

        lo_col, ls_col = st.columns(2)

        def _port_panel(sub, label, accent_color):
            styled_section_label(label)
            items = [
                ("Gross Return",  pct(_g(sub, "return_gross")), delta_color(_g(sub, "return_gross"))),
                ("Net Return",    pct(_g(sub, "return_net")),   delta_color(_g(sub, "return_net"))),
                ("Sharpe (G)",    fmt2(_g(sub, "sharpe_gross")), delta_color(_g(sub, "sharpe_gross"))),
                ("Sharpe (N)",    fmt2(_g(sub, "sharpe_net")),   delta_color(_g(sub, "sharpe_net"))),
                ("Max Drawdown",  pct(_g(sub, "max_dd")),        delta_color(_g(sub, "max_dd"), -0.05)),
                ("Sortino",       fmt2(_g(sub, "sortino_gross")), delta_color(_g(sub, "sortino_gross"))),
                ("Alpha",         fmt_alpha(_g(sub, "alpha")),    delta_color(_g(sub, "alpha"))),
                ("Calmar",        fmt2(_g(sub, "calmar")),        delta_color(_g(sub, "calmar"))),
                ("Avg Turnover",  pct(_g(sub, "avg_turnover")),   TOKENS["text_muted"]),
                ("Beta",          fmt2(_g(sub, "beta")),          TOKENS["text_muted"]),
            ]
            cols = st.columns(2)
            for i, (lbl, val, clr) in enumerate(items):
                with cols[i % 2]:
                    styled_kpi(lbl, val, delta="", delta_color=clr)

        with lo_col:
            _port_panel(lo, "LONG-ONLY (Q5)", TOKENS["accent_primary"])
        with ls_col:
            _port_panel(ls, "LONG / SHORT (Q5 MINUS Q1)", TOKENS["accent_info"])

        styled_divider()
        styled_section_label("3. COST OF TRADING | GROSS vs NET RETURN BY FREQUENCY")
        df_all_f = df_all[(df_all["factor"] == sel_pc_f) & (df_all["weighting"] == sel_pc_wt)]
        fig_gn = go.Figure()
        for pt_val, pt_color, pt_label in [
            ("long_only", TOKENS["accent_primary"], "Long-Only"),
            ("long_short", TOKENS["accent_info"], "Long/Short"),
        ]:
            sub = df_all_f[df_all_f["portfolio_type"] == pt_val]
            g_v = [_safe(sub[sub["frequency"] == f]["return_gross"].values[0])
                   if len(sub[sub["frequency"] == f]) > 0 else np.nan for f in ["monthly","quarterly","annual"]]
            n_v = [_safe(sub[sub["frequency"] == f]["return_net"].values[0])
                   if len(sub[sub["frequency"] == f]) > 0 else np.nan for f in ["monthly","quarterly","annual"]]
            fig_gn.add_trace(go.Scatter(
                x=[f.capitalize() for f in ["monthly","quarterly","annual"]],
                y=[v * 100 if not np.isnan(v) else None for v in g_v],
                name=f"{pt_label} Gross", line=dict(color=pt_color, width=1.5, dash="solid"),
                mode="lines+markers", marker=dict(size=7),
            ))
            fig_gn.add_trace(go.Scatter(
                x=[f.capitalize() for f in ["monthly","quarterly","annual"]],
                y=[v * 100 if not np.isnan(v) else None for v in n_v],
                name=f"{pt_label} Net", line=dict(color=pt_color, width=1.5, dash="dot"),
                mode="lines+markers", marker=dict(size=5),
            ))
        fig_gn.add_hline(y=0, line_color=TOKENS["border_default"], line_width=1)
        fig_gn.update_layout(
            title="Gross vs Net Annualised Return by Frequency",
            xaxis_title="Rebalance Frequency",
            yaxis_title="Annualised Return (%)",
            yaxis_ticksuffix="%",
        )
        apply_plotly_theme(fig_gn)
        st.plotly_chart(fig_gn, use_container_width=True)

        styled_section_label("BEST COMBINATION FOR THIS FACTOR")
        best_fac = df_all[df_all["factor"] == sel_pc_f]
        if not best_fac.empty:
            best_sh  = best_fac.loc[best_fac["sharpe_gross"].idxmax()]
            best_net = best_fac.loc[best_fac["return_net"].idxmax()]
            b1, b2 = st.columns(2)
            with b1:
                styled_card(
                    f"<b style='color:{TOKENS['accent_success']}'>Highest Sharpe.</b> "
                    f"{best_sh['frequency'].capitalize()} | {best_sh['weighting']} | "
                    f"{best_sh['portfolio_type']}. "
                    f"Sharpe {fmt2(_safe(best_sh['sharpe_gross']))} | "
                    f"Return {pct(_safe(best_sh['return_gross']))} | "
                    f"IC {fmt2(_safe(best_sh['mean_ic']))}",
                    accent_color=TOKENS["accent_success"],
                )
            with b2:
                styled_card(
                    f"<b style='color:{TOKENS['accent_primary']}'>Highest Net Return.</b> "
                    f"{best_net['frequency'].capitalize()} | {best_net['weighting']} | "
                    f"{best_net['portfolio_type']}. "
                    f"Net {pct(_safe(best_net['return_net']))} | "
                    f"Sharpe {fmt2(_safe(best_net['sharpe_net']))} | "
                    f"Turnover {pct(_safe(best_net['avg_turnover']))}",
                    accent_color=TOKENS["accent_primary"],
                )


# ======================================================================
# TAB 4 | METHODOLOGY
# ======================================================================

with tab4:
    styled_section_label("FACTOR DEFINITIONS")
    accent = TOKENS["accent_primary"]
    warn = TOKENS["accent_warning"]
    border = TOKENS["border_default"]
    text_pri = TOKENS["text_primary"]
    text_sec = TOKENS["text_secondary"]
    text_mut = TOKENS["text_muted"]
    success = TOKENS["accent_success"]
    danger = TOKENS["accent_danger"]
    bg = TOKENS["bg_surface"]

    st.markdown(f"""
    <div style='background:{bg};border:1px solid {border};border-radius:3px;
                padding:16px 20px;font-size:0.75rem'>
    <table style='width:100%;border-collapse:collapse'>
      <thead><tr style='border-bottom:1px solid {border}'>
        <th style='color:{accent};text-align:left;padding:6px 10px;font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.1em'>Factor</th>
        <th style='color:{accent};text-align:left;padding:6px 10px;font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.1em'>Metric</th>
        <th style='color:{accent};text-align:left;padding:6px 10px;font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.1em'>Direction</th>
        <th style='color:{accent};text-align:left;padding:6px 10px;font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.1em'>Source</th>
        <th style='color:{warn};text-align:left;padding:6px 10px;font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.1em'>PIT Clean?</th>
      </tr></thead>
      <tbody>
        <tr style='border-bottom:1px solid {border}'>
          <td style='padding:6px 10px;color:{text_pri};font-weight:600'>Value</td>
          <td style='padding:6px 10px;color:{text_sec}'>Earnings Yield (1 / PE)</td>
          <td style='padding:6px 10px;color:{success}'>Higher is cheaper</td>
          <td style='padding:6px 10px;color:{text_mut}'>yfinance trailingPE</td>
          <td style='padding:6px 10px;color:{danger}'>No (current only)</td>
        </tr>
        <tr style='border-bottom:1px solid {border}'>
          <td style='padding:6px 10px;color:{text_pri};font-weight:600'>Momentum</td>
          <td style='padding:6px 10px;color:{text_sec}'>12-month return, skip last 1m (12-1)</td>
          <td style='padding:6px 10px;color:{success}'>Higher is stronger trend</td>
          <td style='padding:6px 10px;color:{text_mut}'>Daily close prices</td>
          <td style='padding:6px 10px;color:{success}'>Yes</td>
        </tr>
        <tr style='border-bottom:1px solid {border}'>
          <td style='padding:6px 10px;color:{text_pri};font-weight:600'>Quality</td>
          <td style='padding:6px 10px;color:{text_sec}'>Return on Equity (ROE)</td>
          <td style='padding:6px 10px;color:{success}'>Higher is more profitable</td>
          <td style='padding:6px 10px;color:{text_mut}'>yfinance returnOnEquity</td>
          <td style='padding:6px 10px;color:{danger}'>No (current only)</td>
        </tr>
        <tr style='border-bottom:1px solid {border}'>
          <td style='padding:6px 10px;color:{text_pri};font-weight:600'>Size</td>
          <td style='padding:6px 10px;color:{text_sec}'>log(Market Cap), inverted</td>
          <td style='padding:6px 10px;color:{danger}'>Lower is small cap premium</td>
          <td style='padding:6px 10px;color:{text_mut}'>yfinance marketCap</td>
          <td style='padding:6px 10px;color:{danger}'>No (current only)</td>
        </tr>
        <tr>
          <td style='padding:6px 10px;color:{text_pri};font-weight:600'>Low Vol.</td>
          <td style='padding:6px 10px;color:{text_sec}'>60-day rolling sigma (annualised), inverted</td>
          <td style='padding:6px 10px;color:{danger}'>Lower is better risk-adj.</td>
          <td style='padding:6px 10px;color:{text_mut}'>Daily close prices</td>
          <td style='padding:6px 10px;color:{success}'>Yes</td>
        </tr>
      </tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    m_c1, m_c2 = st.columns(2)
    with m_c1:
        styled_section_label("SIGNAL TIMING: NO LOOKAHEAD")
        styled_card(
            f"At time <i>t</i>, factor scores use <b style='color:{text_pri}'>only data available at t</b>. "
            f"Portfolio is constructed at <i>t</i>. Returns measured <i>t to t+1</i>.<br><br>"
            f"<b style='color:{text_pri}'>Normalisation:</b> Cross-sectional percentile rank [0,1] "
            f"at every rebalance. Size and Low-Vol ranked inversely (lower raw = higher rank).<br><br>"
            f"<b style='color:{text_pri}'>Quintile sort:</b> Q5 = top-ranked. Q1 = lowest-ranked.<br><br>"
            f"<b style='color:{text_pri}'>Long-Only:</b> equal or cap-weighted Q5. Weights to 1.0.<br>"
            f"<b style='color:{text_pri}'>Long/Short:</b> +0.5 Q5, -0.5 Q1. Net zero exposure."
        )

        styled_section_label("TRANSACTION COSTS")
        styled_card(
            f"Net Return = Gross minus Turnover x <b style='color:{text_pri}'>10 bps</b>.<br>"
            f"Turnover = sum of absolute weight change per rebalance.<br>"
            f"Applied only at rebalance dates (monthly, quarterly, annual).<br>"
            f"No market impact, slippage, or bid-ask modelled."
        )

    with m_c2:
        styled_section_label("ANALYTICS DEFINITIONS")
        styled_card(
            f"<b style='color:{text_pri}'>CAGR</b>: geometric compound annual growth.<br>"
            f"<b style='color:{text_pri}'>Sharpe</b>: mean excess return / sigma x sqrt(periods/yr).<br>"
            f"<b style='color:{text_pri}'>Sortino</b>: Sharpe using only downside sigma.<br>"
            f"<b style='color:{text_pri}'>Max DD</b>: worst peak-to-trough (1.0 baseline).<br>"
            f"<b style='color:{text_pri}'>Calmar</b>: Ann. Return / abs(Max DD).<br>"
            f"<b style='color:{text_pri}'>Hit Rate</b>: % periods outperforming SPY (strict &gt;).<br>"
            f"<b style='color:{text_pri}'>IC</b>: Spearman(factor_scores_t, returns_t to t+1).<br>"
            f"<b style='color:{text_pri}'>IC IR</b>: Mean IC / Std(IC), signal consistency.<br>"
            f"<b style='color:{text_pri}'>Alpha / Beta</b>: OLS regression vs SPY (daily returns)."
        )

        styled_section_label("KNOWN LIMITATIONS")
        styled_card(
            f"<b style='color:{warn}'>Survivorship Bias.</b> Universe = today's S&P 500. "
            f"Firms that failed 2014-2026 excluded.<br>"
            f"<b style='color:{warn}'>Non-PIT Fundamentals.</b> PE, ROE, and market cap are "
            f"current values applied historically, not quarterly filings.<br>"
            f"<b style='color:{warn}'>Cap-Weight Not PIT-Clean.</b> Cap-weighted portfolios use "
            f"current market cap for historical weighting.<br>"
            f"<b style='color:{warn}'>No Shorting Costs.</b> L/S portfolios omit borrow costs and margin.<br>"
            f"<b style='color:{warn}'>Risk-Free Rate = 0%.</b> Conservative assumption: raises "
            f"Sharpe vs actual T-bill hurdle.<br>"
            f"<b style='color:{warn}'>No Market Impact.</b> S&P 500 stocks assumed fully liquid at close prices.",
            accent_color=TOKENS["accent_warning"],
        )
