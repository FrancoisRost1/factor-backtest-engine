"""
streamlit_app.py — Factor Model + Backtesting Engine Dashboard

Loads pre-computed results from outputs/backtest_results.csv (summary) and
optional time-series CSVs (outputs/ic_series_*.csv, outputs/quintile_*.csv).

Layout (per CLAUDE.md standards, quant research order):
  TAB 1: OVERVIEW      — filters, best-combination panel, rankings table
  TAB 2: FACTOR ANALYSIS  — Signal Quality to Portfolio Construction to Performance
  TAB 3: PORTFOLIO COMPARISON — long-only vs long/short side-by-side
  TAB 4: METHODOLOGY   — static research write-up

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

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Factor Backtest Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — Bloomberg dark mode ──────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    background-color: #0e1117 !important;
    color: #e8e8e8 !important;
    font-size: 13px !important;
}
h1, h2, h3, h4 {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 2px !important;
    padding: 8px 12px !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    color: #8b949e !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #f0f6fc !important;
}
[data-testid="stDataFrame"] { background: #161b22 !important; }
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e8e8e8 !important;
}
p, span, div, label, small, li { color: #c9d1d9 !important; }
hr { margin: 0.5rem 0 !important; border-color: #30363d !important; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
    background-color: #0e1117 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 0px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #8b949e !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    border-radius: 0px !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #f0f6fc !important;
    border-bottom: 2px solid #58a6ff !important;
    background-color: transparent !important;
}
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

FACTOR_LABELS = {
    "value":          "Value (EY)",
    "momentum":       "Momentum (12-1)",
    "quality":        "Quality (ROE)",
    "size":           "Size (Log MCap)",
    "low_volatility": "Low Vol (60d σ)",
    "composite":      "Composite",
}

# Price-based factors have no point-in-time data issue; fundamental factors do.
FUNDAMENTAL_FACTORS = {"value", "quality", "size"}

ACCENT      = "#58a6ff"
GREEN       = "#3fb950"
RED         = "#f85149"
AMBER       = "#d29922"
PURPLE      = "#bc8cff"
CARD_BG     = "#161b22"
BORDER      = "#30363d"
PLOT_BG     = "#0e1117"
PAPER_BG    = "#0e1117"
GRID_COLOR  = "#21262d"
TEXT_COLOR  = "#8b949e"

QUINTILE_COLORS = ["#f85149", "#d29922", "#8b949e", "#58a6ff", "#3fb950"]  # Q1→Q5

BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color="#c9d1d9", size=11),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=BORDER, tickfont=dict(color=TEXT_COLOR)),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=BORDER, tickfont=dict(color=TEXT_COLOR)),
    margin=dict(l=44, r=20, t=40, b=36),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        font=dict(color="#c9d1d9", size=10),
    ),
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def pct(v, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v * 100:.{dec}f}%"

def fmt_alpha(v):
    """Format annualised alpha, collapsing near-zero to '≈ 0.0%' to avoid '-0.0%'."""
    v = _safe(v)
    if np.isnan(v):
        return ""
    if abs(v) < 0.0005:
        return "≈ 0.0%"
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

def color_val(v, good_above=0.0):
    v = _safe(v)
    if np.isnan(v):
        return TEXT_COLOR
    return GREEN if v > good_above else RED

def apply_layout(fig, title="", yaxis_fmt=None, yaxis2=False):
    layout = dict(**BASE_LAYOUT, title=dict(
        text=title, font=dict(size=11, color=TEXT_COLOR), x=0, xanchor="left"
    ))
    if yaxis_fmt:
        layout["yaxis"] = dict(BASE_LAYOUT["xaxis"], tickformat=yaxis_fmt,
                                gridcolor=GRID_COLOR, linecolor=BORDER,
                                tickfont=dict(color=TEXT_COLOR))
    fig.update_layout(**layout)
    return fig

def kpi_card(label, value, color="#f0f6fc", note=""):
    note_html = f"<div style='font-size:10px;color:{TEXT_COLOR};margin-top:2px'>{note}</div>" if note else ""
    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:2px;
                padding:10px 14px;min-height:72px'>
        <div style='font-size:10px;color:{TEXT_COLOR};text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:4px'>{label}</div>
        <div style='font-size:18px;font-weight:600;color:{color}'>{value}</div>
        {note_html}
    </div>""", unsafe_allow_html=True)

def section_header(title, accent=False):
    border_color = ACCENT if accent else BORDER
    text_color   = "#f0f6fc" if accent else TEXT_COLOR
    st.markdown(
        f"<div style='font-size:10px;font-weight:700;color:{text_color};"
        f"text-transform:uppercase;letter-spacing:0.9px;"
        f"border-bottom:2px solid {border_color};padding-bottom:5px;margin:20px 0 10px 0'>"
        f"{title}</div>",
        unsafe_allow_html=True,
    )

def block_header(number, title, subtitle=""):
    sub_html = f"<span style='font-size:10px;color:{TEXT_COLOR};margin-left:8px'>{subtitle}</span>" if subtitle else ""
    st.markdown(
        f"<div style='background:#161b22;border-left:3px solid {ACCENT};"
        f"padding:8px 12px;margin:24px 0 12px 0;border-radius:0 2px 2px 0'>"
        f"<span style='font-size:11px;font-weight:700;color:{ACCENT}'>{number}. {title.upper()}</span>"
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )

def no_data_banner():
    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {AMBER};border-radius:2px;
                padding:20px 24px;margin:20px 0'>
        <div style='font-size:13px;font-weight:600;color:{AMBER};text-transform:uppercase;letter-spacing:0.8px'>
             No results found
        </div>
        <div style='margin-top:8px;color:#c9d1d9;font-size:12px'>
            Run the backtest pipeline first:<br><br>
            <code style='background:#0d1117;padding:4px 8px;border-radius:2px;
                         color:{ACCENT};font-size:12px'>python main.py</code><br><br>
            This fetches data from Yahoo Finance, runs all 60 backtest combinations,
            and saves results to <code>outputs/backtest_results.csv</code>.
        </div>
    </div>""", unsafe_allow_html=True)

def missing_series_note(label="Per-period data"):
    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {GRID_COLOR};border-radius:2px;"
        f"padding:12px 16px;color:{TEXT_COLOR};font-size:11px'>"
        f"<b style='color:#c9d1d9'>{label} not available.</b><br>"
        f"Run backtest with <code>save_timeseries=True</code> to enable this chart."
        f"</div>", unsafe_allow_html=True)

def interpretation_box():
    st.markdown(f"""
    <div style='background:#161b22;border:1px solid {ACCENT};border-radius:2px;
                padding:14px 18px;margin-top:12px'>
        <div style='font-size:10px;font-weight:700;color:{ACCENT};text-transform:uppercase;
                    letter-spacing:0.8px;margin-bottom:8px'>How to Interpret</div>
        <table style='width:100%;border-collapse:collapse;font-size:11px'>
          <tr>
            <td style='padding:3px 8px;color:{GREEN};font-weight:600;white-space:nowrap'>IC &gt; 0</td>
            <td style='padding:3px 8px;color:#c9d1d9'>Predictive signal: factor rank correlates with forward returns</td>
          </tr>
          <tr>
            <td style='padding:3px 8px;color:{TEXT_COLOR};font-weight:600;white-space:nowrap'>IC ≈ 0</td>
            <td style='padding:3px 8px;color:#c9d1d9'>Weak or absent signal: factor has no predictive power</td>
          </tr>
          <tr>
            <td style='padding:3px 8px;color:{RED};font-weight:600;white-space:nowrap'>IC &lt; 0</td>
            <td style='padding:3px 8px;color:#c9d1d9'>Inverted or biased signal: ranking predicts opposite direction</td>
          </tr>
          <tr>
            <td style='padding:3px 8px;color:{GREEN};font-weight:600;white-space:nowrap'>IC IR &gt; 0.5</td>
            <td style='padding:3px 8px;color:#c9d1d9'>Consistent signal: mean(IC)/std(IC) often considered strong at this level (Grinold &amp; Kahn)</td>
          </tr>
          <tr>
            <td style='padding:3px 8px;color:{GREEN};font-weight:600;white-space:nowrap'>Monotonic Q1 to Q5</td>
            <td style='padding:3px 8px;color:#c9d1d9'>Strong factor: returns increase as predicted across all quintiles</td>
          </tr>
          <tr>
            <td style='padding:3px 8px;color:{TEXT_COLOR};font-weight:600;white-space:nowrap'>Flat quintiles</td>
            <td style='padding:3px 8px;color:#c9d1d9'>No factor signal: ranking explains nothing about returns</td>
          </tr>
        </table>
    </div>""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────

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

# ── Header + research warning ──────────────────────────────────────────────────

st.markdown("""
<div style='border-bottom:1px solid #30363d;padding-bottom:10px;margin-bottom:8px'>
    <span style='font-size:17px;font-weight:700;color:#f0f6fc;letter-spacing:0.5px'>
        FACTOR BACKTEST ENGINE
    </span>
    <span style='font-size:11px;color:#8b949e;margin-left:12px'>
        S&P 500 · 5 Factors · 2016-2026 · 60 Combinations
    </span>
</div>""", unsafe_allow_html=True)

# ⚠ Permanent research-limitation warning
st.markdown(f"""
<div style='background:#1c1200;border:1px solid {AMBER};border-radius:2px;
            padding:10px 16px;margin-bottom:12px;display:flex;align-items:flex-start;gap:10px'>
    <span style='font-size:14px'></span>
    <div>
        <span style='font-size:11px;font-weight:700;color:{AMBER};text-transform:uppercase;
                     letter-spacing:0.6px'>Research Limitation: Results Not Investable</span>
        <span style='font-size:11px;color:#c9a227;margin-left:8px'>
            Fundamental factors (Value, Quality, Size) use <b>non-point-in-time data</b>
           : yfinance current values applied retroactively across the entire backtest period.
            Returns for these factors are likely overstated.
            Price-based factors (Momentum, Low Volatility) are clean.
        </span>
    </div>
</div>""", unsafe_allow_html=True)

df_all = load_results()

tab1, tab2, tab3, tab4 = st.tabs([
    "OVERVIEW",
    "FACTOR ANALYSIS",
    "PORTFOLIO COMPARISON",
    "METHODOLOGY",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if df_all is None:
        no_data_banner()
    else:
        # Filters
        section_header("FILTERS")
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
            # ── Best-performance panel ─────────────────────────────────────────
            st.markdown(f"""
            <div style='margin-top:20px;padding:8px 14px;background:#0d1117;
                        border:1px solid {BORDER};border-radius:2px;margin-bottom:2px'>
                <span style='font-size:10px;font-weight:700;color:{AMBER};
                             text-transform:uppercase;letter-spacing:0.8px'>
                    ▶ BEST PERFORMANCE (by Gross Sharpe): not best signal quality
                </span>
            </div>
            <div style='padding:4px 14px 8px 14px;font-size:10px;color:{TEXT_COLOR}'>
                Selected by Sharpe ratio: may reflect risk exposure or data bias,
                not genuine predictive power. Check IC and quintile spread to assess signal quality.
            </div>""", unsafe_allow_html=True)

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
                kpi_card("Ann. Return (Gross)", pct(ret_v), color=color_val(ret_v))
            with k2:
                kpi_card("Sharpe (Gross)", fmt2(sharpe_v), color=color_val(sharpe_v))
            with k3:
                kpi_card("Max Drawdown", pct(mdd_v), color=color_val(mdd_v, -0.05))
            with k4:
                kpi_card("Alpha (Ann.)", fmt_alpha(alpha_v), color=color_val(alpha_v))
            with k5:
                kpi_card("Mean IC", fmt2(ic_v), color=color_val(ic_v),
                         note=f"IR: {fmt2(ic_ir_v)}")
            with k6:
                kpi_card("Avg Turnover / Period", pct(to_v), color="#f0f6fc")

            st.markdown(
                f"<div style='font-size:10px;color:{TEXT_COLOR};margin:6px 0 16px 2px'>"
                f"Combination: <b style='color:#f0f6fc'>"
                f"{FACTOR_LABELS.get(best['factor'], best['factor'])}</b> · "
                f"{best['frequency']} · {best['weighting']} · {best['portfolio_type']}"
                f"</div>", unsafe_allow_html=True)

            # ── Summary table ──────────────────────────────────────────────────
            section_header("ALL COMBINATIONS: SORTED BY SHARPE (GROSS)")

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
            # Alpha: use fmt_alpha to suppress "-0.0%"
            if "Alpha" in df_show.columns:
                df_show["Alpha"] = df_show["Alpha"].apply(
                    lambda v: fmt_alpha(_safe(v))
                )
            # Sharpe / Sortino / Calmar / Beta / R²: 2 decimals
            for col in ["Sharpe (G)", "Sharpe (N)", "Sortino", "Calmar", "Beta"]:
                if col in df_show.columns:
                    df_show[col] = df_show[col].apply(
                        lambda v: fmt2(_safe(v)) if not (isinstance(v, float) and np.isnan(v)) else ""
                    )
            # Mean IC and IC IR: 3 decimals for precision
            for col in ["Mean IC", "IC IR"]:
                if col in df_show.columns:
                    df_show[col] = df_show[col].apply(
                        lambda v: fmt3(_safe(v)) if not (isinstance(v, float) and np.isnan(v)) else ""
                    )
            st.dataframe(df_show, use_container_width=True,
                         height=min(600, 38 + 35 * len(df_show)), hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FACTOR ANALYSIS
# Structure: 1. Signal Quality to 2. Portfolio Construction to 3. Performance
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if df_all is None:
        no_data_banner()
    else:
        # ── Selection controls ─────────────────────────────────────────────────
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

        # Warn if fundamental factor selected
        if sel_fa in FUNDAMENTAL_FACTORS:
            st.markdown(f"""
            <div style='background:#1c1200;border:1px solid {AMBER};border-radius:2px;
                        padding:8px 14px;margin:4px 0 8px 0;font-size:11px;color:#c9a227'>
                 <b style='color:{AMBER}'>{FACTOR_LABELS.get(sel_fa, sel_fa)}</b>
                uses non-point-in-time fundamental data. Signal metrics and performance
                figures are likely upward-biased.
            </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # BLOCK 1 — SIGNAL QUALITY
        # ══════════════════════════════════════════════════════════════════════
        block_header("1", "Signal Quality", "Does the factor predict returns?")

        sig_left, sig_right = st.columns([5, 2])

        with sig_left:
            # IC statistics cards
            section_header("INFORMATION COEFFICIENT", accent=True)
            ic_mean  = _safe(row_sel["mean_ic"])  if row_sel is not None else np.nan
            ic_ir    = _safe(row_sel["ic_ir"])    if row_sel is not None else np.nan

            ic_c1, ic_c2, ic_c3 = st.columns(3)
            with ic_c1:
                ic_col = GREEN if not np.isnan(ic_mean) and ic_mean > 0.02 else \
                         RED   if not np.isnan(ic_mean) and ic_mean < 0 else AMBER
                kpi_card("Mean IC", fmt2(ic_mean), color=ic_col)
            with ic_c2:
                irir_col = GREEN if not np.isnan(ic_ir) and ic_ir > 0.5 else \
                           AMBER if not np.isnan(ic_ir) and ic_ir > 0 else RED
                kpi_card("IC IR  [mean(IC) / std(IC)]", fmt2(ic_ir), color=irir_col,
                         note="Reference: > 0.5 often considered strong")
            with ic_c3:
                # Computed from actual per-period IC series only — not from summary stats
                _ic_ts_for_pct = load_ic_series(sel_fa, sel_fa_freq)
                if _ic_ts_for_pct is not None and len(_ic_ts_for_pct) > 0:
                    _col = _ic_ts_for_pct.columns[0]
                    _vals = _ic_ts_for_pct[_col].dropna()
                    if len(_vals) > 0:
                        pct_pos = float((_vals > 0).mean())
                        kpi_card("% Periods IC > 0", pct(pct_pos),
                                 color=color_val(pct_pos, 0.5),
                                 note=f"n = {len(_vals)} periods")
                    else:
                        kpi_card("% Periods IC > 0", "N/A: insufficient data",
                                 color=TEXT_COLOR)
                else:
                    kpi_card("% Periods IC > 0", "N/A: series unavailable",
                             color=TEXT_COLOR,
                             note="save_timeseries=True required")

            # IC time series — load once, reuse for "% IC > 0" card
            section_header("IC OVER TIME")
            ic_ts = load_ic_series(sel_fa, sel_fa_freq)
            show_rolling = st.checkbox("Show 6-period rolling mean", value=True, key="ic_rolling")

            if ic_ts is not None and len(ic_ts) > 0:
                col_name = ic_ts.columns[0] if len(ic_ts.columns) > 0 else None
                if col_name is not None:
                    ic_vals = ic_ts[col_name].dropna()
                    fig_ic = go.Figure()
                    fig_ic.add_trace(go.Bar(
                        x=ic_vals.index, y=ic_vals.values,
                        marker_color=[GREEN if v > 0 else RED for v in ic_vals.values],
                        marker_line_width=0, opacity=0.8, name="IC",
                        showlegend=True,
                    ))
                    if show_rolling and len(ic_vals) >= 6:
                        roll = ic_vals.rolling(6).mean()
                        fig_ic.add_trace(go.Scatter(
                            x=roll.index, y=roll.values,
                            line=dict(color=AMBER, width=2), name="6-period MA",
                        ))
                    fig_ic.add_hline(y=0, line_color=BORDER, line_width=1.5)
                    apply_layout(fig_ic, f"{FACTOR_LABELS.get(sel_fa, sel_fa)}: IC per Period")
                    st.plotly_chart(fig_ic, use_container_width=True)
            else:
                missing_series_note("IC time-series")
                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

            # Quintile return spread
            section_header("QUINTILE RETURN SPREAD (Q1 = WORST, Q5 = BEST)")
            qt_data = load_quintile_returns(sel_fa, sel_fa_freq, sel_fa_wt)

            if qt_data is not None and len(qt_data) > 0:
                # Compute mean return per quintile
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
                    textfont=dict(size=11, color="#c9d1d9"),
                    showlegend=False,
                ))
                fig_qbar.add_hline(y=0, line_color=BORDER, line_width=1)
                apply_layout(fig_qbar,
                    f"Mean Period Return by Quintile: {sel_fa_freq.capitalize()}")
                fig_qbar.update_layout(yaxis_ticksuffix="%")
                st.plotly_chart(fig_qbar, use_container_width=True)

                # Cumulative return by quintile
                section_header("CUMULATIVE RETURN BY QUINTILE")
                log_scale = st.checkbox("Log scale", value=False, key="log_q")
                fig_qcum = go.Figure()
                for i, col in enumerate(qt_data.columns):
                    cum = (1 + qt_data[col].fillna(0)).cumprod()
                    label = f"Q{int(col)}" if str(col).isdigit() else str(col)
                    fig_qcum.add_trace(go.Scatter(
                        x=cum.index, y=cum.values,
                        name=label,
                        line=dict(color=QUINTILE_COLORS[i % 5], width=2),
                        mode="lines",
                    ))
                apply_layout(fig_qcum,
                    f"Cumulative Return by Quintile: {sel_fa_freq.capitalize()}")
                if log_scale:
                    fig_qcum.update_layout(yaxis_type="log")
                st.plotly_chart(fig_qcum, use_container_width=True)
            else:
                missing_series_note("Quintile return series")
                st.markdown(
                    f"<div style='background:{CARD_BG};border:1px solid {GRID_COLOR};"
                    f"border-radius:2px;padding:12px 16px;color:{TEXT_COLOR};font-size:11px;"
                    f"margin-top:6px'>Cumulative quintile chart also requires per-period data.</div>",
                    unsafe_allow_html=True)

        with sig_right:
            st.markdown("<div style='margin-top:48px'></div>", unsafe_allow_html=True)
            interpretation_box()

            # IC across frequencies summary
            section_header("IC BY FREQUENCY")
            for f in ["monthly", "quarterly", "annual"]:
                sub = df_fa[(df_fa["frequency"] == f) & (df_fa["weighting"] == sel_fa_wt)]
                if len(sub) > 0:
                    ic_v = _safe(sub["mean_ic"].values[0])
                    ir_v = _safe(sub["ic_ir"].values[0])
                    ic_col = GREEN if not np.isnan(ic_v) and ic_v > 0 else RED
                    st.markdown(
                        f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
                        f"border-radius:2px;padding:8px 12px;margin-bottom:6px'>"
                        f"<div style='font-size:10px;color:{TEXT_COLOR};text-transform:uppercase;letter-spacing:0.5px'>"
                        f"{f.capitalize()}</div>"
                        f"<div style='font-size:14px;font-weight:600;color:{ic_col}'>"
                        f"IC {fmt2(ic_v)}</div>"
                        f"<div style='font-size:10px;color:{TEXT_COLOR}'>IR: {fmt2(ir_v)}</div>"
                        f"</div>", unsafe_allow_html=True)

        st.markdown(f"<div style='border-top:1px solid {BORDER};margin:24px 0 0 0'></div>",
                    unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # BLOCK 2 — PORTFOLIO CONSTRUCTION
        # ══════════════════════════════════════════════════════════════════════
        block_header("2", "Portfolio Construction",
                     "How the signal translates into a portfolio")

        pc_c1, pc_c2 = st.columns([3, 2])

        with pc_c1:
            section_header("SHARPE RATIO: EQUAL vs CAP-WEIGHT BY FREQUENCY")
            freqs = ["monthly", "quarterly", "annual"]
            wts   = ["equal", "cap_weight"]
            wt_colors = {"equal": ACCENT, "cap_weight": PURPLE}

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
                    marker_line_color=BORDER, marker_line_width=1,
                    text=[fmt2(v) for v in sh_vals],
                    textposition="outside",
                    textfont=dict(size=10, color="#c9d1d9"),
                ))
            fig_sh.add_hline(y=0, line_color=BORDER, line_width=1)
            apply_layout(fig_sh, "Sharpe (Gross)")
            fig_sh.update_layout(barmode="group", bargap=0.2)
            st.plotly_chart(fig_sh, use_container_width=True)

        with pc_c2:
            section_header("TURNOVER BY FREQUENCY")
            for wt_val, wt_color, wt_label in [
                ("equal",      ACCENT,  "Equal Weight"),
                ("cap_weight", PURPLE,  "Cap Weight"),
            ]:
                sub_wt = df_fa[df_fa["weighting"] == wt_val]
                st.markdown(
                    f"<div style='font-size:11px;font-weight:600;color:{wt_color};"
                    f"margin:10px 0 4px 0'>{wt_label}</div>",
                    unsafe_allow_html=True)
                to_cols = st.columns(3)
                for i, f in enumerate(freqs):
                    sub_f = sub_wt[sub_wt["frequency"] == f]
                    to_v = _safe(sub_f["avg_turnover"].values[0]) if len(sub_f) > 0 else np.nan
                    with to_cols[i]:
                        kpi_card(f.capitalize(), pct(to_v), color="#f0f6fc")

        # Equal-weight vs cap-weight detail table
        section_header("EQUAL vs CAP-WEIGHT: FULL COMPARISON TABLE")
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

        st.markdown(f"<div style='border-top:1px solid {BORDER};margin:24px 0 0 0'></div>",
                    unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # BLOCK 3 — PERFORMANCE
        # ══════════════════════════════════════════════════════════════════════
        block_header("3", "Performance",
                     "Returns, risk-adjusted metrics, and factor regression")

        show_net = st.checkbox("Compare gross vs net", value=True, key="show_net")

        perf_c1, perf_c2 = st.columns([3, 2])

        with perf_c1:
            section_header("ANNUALISED RETURN BY FREQUENCY")
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
            fig_ret.add_hline(y=0, line_color=BORDER, line_width=1)
            apply_layout(fig_ret, "Annualised Return (%)")
            fig_ret.update_layout(barmode="group", yaxis_ticksuffix="%")
            st.plotly_chart(fig_ret, use_container_width=True)

        with perf_c2:
            section_header("KEY PERFORMANCE METRICS")
            if row_sel is not None:
                metrics = [
                    ("Ann. Return (Gross)", pct(_safe(row_sel.get("return_gross", np.nan))),
                     color_val(_safe(row_sel.get("return_gross", np.nan)))),
                    ("Ann. Return (Net)",   pct(_safe(row_sel.get("return_net", np.nan))),
                     color_val(_safe(row_sel.get("return_net", np.nan)))),
                    ("Sharpe (Gross)",      fmt2(_safe(row_sel.get("sharpe_gross", np.nan))),
                     color_val(_safe(row_sel.get("sharpe_gross", np.nan)))),
                    ("Sortino",             fmt2(_safe(row_sel.get("sortino_gross", np.nan))),
                     color_val(_safe(row_sel.get("sortino_gross", np.nan)))),
                    ("Max Drawdown",        pct(_safe(row_sel.get("max_dd", np.nan))),
                     color_val(_safe(row_sel.get("max_dd", np.nan)), -0.05)),
                    ("Calmar",              fmt2(_safe(row_sel.get("calmar", np.nan))),
                     color_val(_safe(row_sel.get("calmar", np.nan)))),
                    ("Hit Rate vs SPY",     pct(_safe(row_sel.get("hit_rate", np.nan))),
                     color_val(_safe(row_sel.get("hit_rate", np.nan)), 0.5)),
                    ("Alpha (Ann.)",        fmt_alpha(_safe(row_sel.get("alpha", np.nan))),
                     color_val(_safe(row_sel.get("alpha", np.nan)))),
                    ("Beta vs SPY",         fmt2(_safe(row_sel.get("beta", np.nan))),
                     "#f0f6fc"),
                    ("R²",                  fmt2(_safe(row_sel.get("r_squared", np.nan))),
                     "#f0f6fc"),
                ]
                m_c1, m_c2 = st.columns(2)
                for i, (lbl, val, clr) in enumerate(metrics):
                    with (m_c1 if i % 2 == 0 else m_c2):
                        kpi_card(lbl, val, color=clr)
                        st.markdown("<div style='margin:4px 0'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if df_all is None:
        no_data_banner()
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

        # IC first — signal quality header
        block_header("1", "Signal Quality (shared across portfolio types)")
        ic_lo = _g(lo, "mean_ic")
        icir_lo = _g(lo, "ic_ir")
        sq_c1, sq_c2, sq_c3, sq_c4 = st.columns(4)
        with sq_c1:
            kpi_card("Mean IC", fmt2(ic_lo),
                     color=color_val(ic_lo), note="(same for both types)")
        with sq_c2:
            kpi_card("IC IR", fmt2(icir_lo),
                     color=GREEN if not np.isnan(icir_lo) and icir_lo > 0.5 else
                           AMBER if not np.isnan(icir_lo) and icir_lo > 0 else RED)
        with sq_c3:
            kpi_card("Hit Rate (Long-Only)", pct(_g(lo, "hit_rate")),
                     color=color_val(_g(lo, "hit_rate"), 0.5))
        with sq_c4:
            kpi_card("Hit Rate (L/S)", pct(_g(ls, "hit_rate")),
                     color=color_val(_g(ls, "hit_rate"), 0.5))

        # Side-by-side portfolio comparison
        block_header("2", "Long-Only vs Long/Short",
                     "Q5 only vs Q5 long + Q1 short (dollar-neutral)")

        lo_col, ls_col = st.columns(2)

        def _port_panel(sub, label, accent_color):
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;color:{accent_color};"
                f"padding:6px 0;border-bottom:2px solid {accent_color};margin-bottom:10px'>"
                f"{label}</div>", unsafe_allow_html=True)
            items = [
                ("Gross Return",  pct(_g(sub, "return_gross")), color_val(_g(sub, "return_gross"))),
                ("Net Return",    pct(_g(sub, "return_net")),   color_val(_g(sub, "return_net"))),
                ("Sharpe (G)",    fmt2(_g(sub, "sharpe_gross")), color_val(_g(sub, "sharpe_gross"))),
                ("Sharpe (N)",    fmt2(_g(sub, "sharpe_net")),   color_val(_g(sub, "sharpe_net"))),
                ("Max Drawdown",  pct(_g(sub, "max_dd")),        color_val(_g(sub, "max_dd"), -0.05)),
                ("Sortino",       fmt2(_g(sub, "sortino_gross")), color_val(_g(sub, "sortino_gross"))),
                ("Alpha",         fmt_alpha(_g(sub, "alpha")),    color_val(_g(sub, "alpha"))),
                ("Calmar",        fmt2(_g(sub, "calmar")),        color_val(_g(sub, "calmar"))),
                ("Avg Turnover",  pct(_g(sub, "avg_turnover")),   "#f0f6fc"),
                ("Beta",          fmt2(_g(sub, "beta")),          "#f0f6fc"),
            ]
            cols = st.columns(2)
            for i, (lbl, val, clr) in enumerate(items):
                with cols[i % 2]:
                    kpi_card(lbl, val, color=clr)
                    st.markdown("<div style='margin:4px 0'></div>", unsafe_allow_html=True)

        with lo_col:
            _port_panel(lo, "LONG-ONLY (Q5)", ACCENT)
        with ls_col:
            _port_panel(ls, "LONG / SHORT (Q5 − Q1)", PURPLE)

        # Gross vs net cost of trading
        block_header("3", "Cost of Trading",
                     "Gross vs net return: cost increases with rebalancing frequency")
        df_all_f = df_all[(df_all["factor"] == sel_pc_f) & (df_all["weighting"] == sel_pc_wt)]
        fig_gn = go.Figure()
        for pt_val, pt_color, pt_label in [("long_only", ACCENT, "Long-Only"),
                                            ("long_short", PURPLE, "Long/Short")]:
            sub = df_all_f[df_all_f["portfolio_type"] == pt_val]
            g_v = [_safe(sub[sub["frequency"] == f]["return_gross"].values[0])
                   if len(sub[sub["frequency"] == f]) > 0 else np.nan for f in ["monthly","quarterly","annual"]]
            n_v = [_safe(sub[sub["frequency"] == f]["return_net"].values[0])
                   if len(sub[sub["frequency"] == f]) > 0 else np.nan for f in ["monthly","quarterly","annual"]]
            fig_gn.add_trace(go.Scatter(
                x=[f.capitalize() for f in ["monthly","quarterly","annual"]],
                y=[v * 100 if not np.isnan(v) else None for v in g_v],
                name=f"{pt_label} Gross", line=dict(color=pt_color, width=2.5, dash="solid"),
                mode="lines+markers", marker=dict(size=7),
            ))
            fig_gn.add_trace(go.Scatter(
                x=[f.capitalize() for f in ["monthly","quarterly","annual"]],
                y=[v * 100 if not np.isnan(v) else None for v in n_v],
                name=f"{pt_label} Net", line=dict(color=pt_color, width=2, dash="dot"),
                mode="lines+markers", marker=dict(size=5),
            ))
        fig_gn.add_hline(y=0, line_color=BORDER, line_width=1)
        apply_layout(fig_gn, "Gross vs Net Annualised Return by Frequency (%)")
        fig_gn.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(fig_gn, use_container_width=True)

        # Best combination
        section_header("BEST COMBINATION FOR THIS FACTOR")
        best_fac = df_all[df_all["factor"] == sel_pc_f]
        if not best_fac.empty:
            best_sh  = best_fac.loc[best_fac["sharpe_gross"].idxmax()]
            best_net = best_fac.loc[best_fac["return_net"].idxmax()]
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(
                    f"<div style='background:{CARD_BG};border:1px solid {GREEN};"
                    f"border-radius:2px;padding:12px 16px'>"
                    f"<div style='font-size:10px;color:{TEXT_COLOR};text-transform:uppercase;"
                    f"letter-spacing:0.5px;margin-bottom:6px'>Highest Sharpe</div>"
                    f"<div style='font-size:13px;font-weight:600;color:{GREEN}'>"
                    f"{best_sh['frequency'].capitalize()} · {best_sh['weighting']} · {best_sh['portfolio_type']}</div>"
                    f"<div style='font-size:11px;color:#c9d1d9;margin-top:4px'>"
                    f"Sharpe: {fmt2(_safe(best_sh['sharpe_gross']))} · "
                    f"Return: {pct(_safe(best_sh['return_gross']))} · "
                    f"IC: {fmt2(_safe(best_sh['mean_ic']))}</div></div>",
                    unsafe_allow_html=True)
            with b2:
                st.markdown(
                    f"<div style='background:{CARD_BG};border:1px solid {ACCENT};"
                    f"border-radius:2px;padding:12px 16px'>"
                    f"<div style='font-size:10px;color:{TEXT_COLOR};text-transform:uppercase;"
                    f"letter-spacing:0.5px;margin-bottom:6px'>Highest Net Return</div>"
                    f"<div style='font-size:13px;font-weight:600;color:{ACCENT}'>"
                    f"{best_net['frequency'].capitalize()} · {best_net['weighting']} · {best_net['portfolio_type']}</div>"
                    f"<div style='font-size:11px;color:#c9d1d9;margin-top:4px'>"
                    f"Net: {pct(_safe(best_net['return_net']))} · "
                    f"Sharpe: {fmt2(_safe(best_net['sharpe_net']))} · "
                    f"Turnover: {pct(_safe(best_net['avg_turnover']))}</div></div>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    section_header("FACTOR DEFINITIONS")
    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:2px;
                padding:16px 20px;font-size:12px'>
    <table style='width:100%;border-collapse:collapse'>
      <thead><tr style='border-bottom:1px solid {BORDER}'>
        <th style='color:{ACCENT};text-align:left;padding:6px 10px;font-size:10px;
                   text-transform:uppercase;letter-spacing:0.5px'>Factor</th>
        <th style='color:{ACCENT};text-align:left;padding:6px 10px;font-size:10px;
                   text-transform:uppercase;letter-spacing:0.5px'>Metric</th>
        <th style='color:{ACCENT};text-align:left;padding:6px 10px;font-size:10px;
                   text-transform:uppercase;letter-spacing:0.5px'>Direction</th>
        <th style='color:{ACCENT};text-align:left;padding:6px 10px;font-size:10px;
                   text-transform:uppercase;letter-spacing:0.5px'>Source</th>
        <th style='color:{AMBER};text-align:left;padding:6px 10px;font-size:10px;
                   text-transform:uppercase;letter-spacing:0.5px'>PIT Clean?</th>
      </tr></thead>
      <tbody>
        <tr style='border-bottom:1px solid {GRID_COLOR}'>
          <td style='padding:6px 10px;color:#f0f6fc;font-weight:600'>Value</td>
          <td style='padding:6px 10px;color:#c9d1d9'>Earnings Yield (1 / PE)</td>
          <td style='padding:6px 10px;color:{GREEN}'>Higher to Cheaper</td>
          <td style='padding:6px 10px;color:{TEXT_COLOR}'>yfinance trailingPE</td>
          <td style='padding:6px 10px;color:{RED}'>No (current only)</td>
        </tr>
        <tr style='border-bottom:1px solid {GRID_COLOR}'>
          <td style='padding:6px 10px;color:#f0f6fc;font-weight:600'>Momentum</td>
          <td style='padding:6px 10px;color:#c9d1d9'>12-month return, skip last 1m (12-1)</td>
          <td style='padding:6px 10px;color:{GREEN}'>Higher to Stronger trend</td>
          <td style='padding:6px 10px;color:{TEXT_COLOR}'>Daily close prices</td>
          <td style='padding:6px 10px;color:{GREEN}'>Yes</td>
        </tr>
        <tr style='border-bottom:1px solid {GRID_COLOR}'>
          <td style='padding:6px 10px;color:#f0f6fc;font-weight:600'>Quality</td>
          <td style='padding:6px 10px;color:#c9d1d9'>Return on Equity (ROE)</td>
          <td style='padding:6px 10px;color:{GREEN}'>Higher to More profitable</td>
          <td style='padding:6px 10px;color:{TEXT_COLOR}'>yfinance returnOnEquity</td>
          <td style='padding:6px 10px;color:{RED}'>No (current only)</td>
        </tr>
        <tr style='border-bottom:1px solid {GRID_COLOR}'>
          <td style='padding:6px 10px;color:#f0f6fc;font-weight:600'>Size</td>
          <td style='padding:6px 10px;color:#c9d1d9'>log(Market Cap): inverted</td>
          <td style='padding:6px 10px;color:{RED}'>Lower to Small cap premium</td>
          <td style='padding:6px 10px;color:{TEXT_COLOR}'>yfinance marketCap</td>
          <td style='padding:6px 10px;color:{RED}'>No (current only)</td>
        </tr>
        <tr>
          <td style='padding:6px 10px;color:#f0f6fc;font-weight:600'>Low Vol.</td>
          <td style='padding:6px 10px;color:#c9d1d9'>60-day rolling σ (annualised): inverted</td>
          <td style='padding:6px 10px;color:{RED}'>Lower to Better risk-adj.</td>
          <td style='padding:6px 10px;color:{TEXT_COLOR}'>Daily close prices</td>
          <td style='padding:6px 10px;color:{GREEN}'>Yes</td>
        </tr>
      </tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    m_c1, m_c2 = st.columns(2)
    with m_c1:
        section_header("SIGNAL TIMING: NO LOOKAHEAD")
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:2px;
                    padding:14px 16px;font-size:12px;line-height:1.9;color:#c9d1d9'>
        At time <i>t</i>, factor scores use <b style='color:#f0f6fc'>only data available at t</b>.
        Portfolio is constructed at <i>t</i>. Returns measured <i>t to t+1</i>.<br><br>
        <b style='color:#f0f6fc'>Normalisation:</b> Cross-sectional percentile rank [0,1]
        at every rebalance. Size and Low-Vol ranked inversely (lower raw = higher rank).<br><br>
        <b style='color:#f0f6fc'>Quintile sort:</b> Q5 = top-ranked. Q1 = lowest-ranked.<br><br>
        <b style='color:#f0f6fc'>Long-Only:</b> equal or cap-weighted Q5. Weights to 1.0<br>
        <b style='color:#f0f6fc'>Long/Short:</b> +0.5 Q5, −0.5 Q1. Net zero exposure.
        </div>""", unsafe_allow_html=True)

        section_header("TRANSACTION COSTS")
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:2px;
                    padding:14px 16px;font-size:12px;line-height:1.9;color:#c9d1d9'>
        Net Return = Gross − Turnover × <b style='color:#f0f6fc'>10 bps</b><br>
        Turnover = Σ|Δweight| per rebalance<br>
        Applied only at rebalance dates (monthly / quarterly / annual)<br>
        No market impact, slippage, or bid-ask modelled.
        </div>""", unsafe_allow_html=True)

    with m_c2:
        section_header("ANALYTICS DEFINITIONS")
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:2px;
                    padding:14px 16px;font-size:12px;line-height:1.9;color:#c9d1d9'>
        <b style='color:#f0f6fc'>CAGR</b>: geometric compound annual growth<br>
        <b style='color:#f0f6fc'>Sharpe</b>: mean excess return / σ × √periods/yr<br>
        <b style='color:#f0f6fc'>Sortino</b>: Sharpe using only downside σ<br>
        <b style='color:#f0f6fc'>Max DD</b>: worst peak-to-trough (with 1.0 baseline)<br>
        <b style='color:#f0f6fc'>Calmar</b>: Ann. Return / |Max DD|<br>
        <b style='color:#f0f6fc'>Hit Rate</b>: % periods outperforming SPY (strict &gt;)<br>
        <b style='color:#f0f6fc'>IC</b>: Spearman(factor_scores_t, returns_t to t+1)<br>
        <b style='color:#f0f6fc'>IC IR</b>: Mean IC / Std(IC): signal consistency<br>
        <b style='color:#f0f6fc'>Alpha / Beta</b>: OLS regression vs SPY (daily returns)
        </div>""", unsafe_allow_html=True)

        section_header("KNOWN LIMITATIONS")
        st.markdown(f"""
        <div style='background:#1c1200;border:1px solid {AMBER};border-radius:2px;
                    padding:14px 16px;font-size:12px;line-height:1.9;color:#c9a227'>
        <b style='color:{AMBER}'>Survivorship Bias.</b>
        <span style='color:#c9d1d9'> Universe = today's S&P 500. Firms that failed 2014-2026 excluded.</span><br>
        <b style='color:{AMBER}'>Non-PIT Fundamentals.</b>
        <span style='color:#c9d1d9'> PE, ROE, and market cap are current values applied historically: not quarterly filings.</span><br>
        <b style='color:{AMBER}'>Cap-Weight Not PIT-Clean.</b>
        <span style='color:#c9d1d9'> Cap-weighted portfolios use current market cap for historical weighting.
        Historical market cap at each rebalance date would be required for a clean backtest.</span><br>
        <b style='color:{AMBER}'>No Shorting Costs.</b>
        <span style='color:#c9d1d9'> L/S portfolios omit borrow costs and margin.</span><br>
        <b style='color:{AMBER}'>Risk-Free Rate = 0%.</b>
        <span style='color:#c9d1d9'> Conservative assumption: raises Sharpe vs actual T-bill hurdle.</span><br>
        <b style='color:{AMBER}'>No Market Impact.</b>
        <span style='color:#c9d1d9'> S&P 500 stocks assumed fully liquid at close prices.</span>
        </div>""", unsafe_allow_html=True)
